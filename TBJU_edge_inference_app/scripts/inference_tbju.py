#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

IMG_SIZE = (640, 640)  # width, height
OCR_SIZE = (384, 32)   # width, height

DEFAULT_CLASSES = [
    'TBJU_region',
    'carriage_rim_region',
    'carriage_rim_debris',
    'track_region',
    'track_intrusion_debris',
    'door_region',
]

CLASS_COLORS = {
    0: (0, 220, 0),
    1: (255, 180, 0),
    2: (0, 128, 255),
    3: (255, 0, 160),
    4: (0, 0, 255),
    5: (180, 80, 255),
}


def load_chars(path):
    chars = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            s = line.strip()
            if s:
                chars.append(s)
    if not chars:
        raise ValueError(f'empty char dict: {path}')
    return chars


def load_classes(path):
    if not path or not Path(path).exists():
        return DEFAULT_CLASSES
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                names.append(parts[1])
            else:
                names.append(s)
    return names or DEFAULT_CLASSES


class RKNNRunner:
    def __init__(self, model_path, use_all_cores=False):
        self.model_path = str(model_path)
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f'load_rknn failed: {self.model_path}, ret={ret}')
        if use_all_cores:
            try:
                ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            except Exception:
                ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f'init_runtime failed: {self.model_path}, ret={ret}')

    def run(self, x, data_format=None):
        if data_format is None:
            outputs = self.rknn.inference(inputs=[x])
        else:
            outputs = self.rknn.inference(inputs=[x], data_format=data_format)
        if outputs is None:
            raise RuntimeError(f'inference returned None: {self.model_path}')
        return outputs

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


def letterbox_bgr(image, new_shape=IMG_SIZE, color=(0, 0, 0)):
    src_h, src_w = image.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / src_w, new_h / src_h)
    resized_w = int(round(src_w * r))
    resized_h = int(round(src_h * r))
    dw = (new_w - resized_w) / 2.0
    dh = (new_h - resized_h) / 2.0

    if (src_w, src_h) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)


def nms_xyxy(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        denom = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / denom
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def decode_yolov8(outputs, ratio, pad, orig_shape, conf_thresh, iou_thresh, max_det=100):
    pred = np.asarray(outputs[0])
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError(f'unexpected YOLO output shape: {outputs[0].shape}')
    if pred.shape[0] <= 20 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
    if pred.shape[1] < 5:
        raise RuntimeError(f'unexpected YOLO output layout: {pred.shape}')

    boxes_xywh = pred[:, :4].astype(np.float32)
    cls_scores = pred[:, 4:].astype(np.float32)
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    mask = scores >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if len(scores) == 0:
        return []

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    orig_h, orig_w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    dets = []
    for cls_id in sorted(set(class_ids.tolist())):
        idxs = np.where(class_ids == cls_id)[0]
        keep = nms_xyxy(boxes[idxs], scores[idxs], iou_thresh)
        for k in keep:
            i = idxs[k]
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({
                'bbox': [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                'score': float(scores[i]),
                'class_id': int(class_ids[i]),
            })
    dets.sort(key=lambda d: d['score'], reverse=True)
    return dets[:max_det]


def preprocess_yolo(image_bgr):
    padded, ratio, pad = letterbox_bgr(image_bgr, IMG_SIZE, color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8), ratio, pad


def preprocess_ocr(crop_bgr):
    if crop_bgr.size == 0:
        raise ValueError('empty OCR crop')
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, OCR_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32)


def ctc_decode(logits, chars, blank_idx=0):
    arr = np.asarray(logits)
    arr = np.squeeze(arr)
    num_classes = len(chars) + 1
    if arr.ndim != 2:
        raise RuntimeError(f'unexpected OCR output shape: {logits.shape}')
    if arr.shape[-1] != num_classes and arr.shape[0] == num_classes:
        arr = arr.T
    ids = np.argmax(arr, axis=-1).tolist()
    result = []
    prev = None
    for idx in ids:
        if idx != blank_idx and idx != prev and 1 <= idx <= len(chars):
            result.append(chars[idx - 1])
        prev = idx
    return ''.join(result)


def expand_box(box, pad, shape):
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad),
    ]


def class_name(class_id, classes):
    if 0 <= class_id < len(classes):
        return classes[class_id]
    return f'class_{class_id}'


def run_one(image_path, yolo, ocr, chars, classes, args):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'failed to read image: {image_path}')

    t0 = time.time()
    yolo_input, ratio, pad = preprocess_yolo(image)
    yolo_outputs = yolo.run(yolo_input)
    dets = decode_yolov8(yolo_outputs, ratio, pad, image.shape, args.conf, args.iou, args.max_det)
    t1 = time.time()

    rows = []
    vis = image.copy()
    ocr_texts = []
    for idx, det in enumerate(dets):
        cls_id = det['class_id']
        name = class_name(cls_id, classes)
        box = expand_box(det['bbox'], args.crop_pad if cls_id == args.tbju_class_id else 0, image.shape)
        x1, y1, x2, y2 = box
        text = ''
        if (not args.skip_ocr) and cls_id == args.tbju_class_id and ocr is not None:
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                ocr_input = preprocess_ocr(crop)
                # OCR input is NCHW. Do not remove data_format='nchw'.
                ocr_outputs = ocr.run(ocr_input, data_format='nchw')
                text = ctc_decode(ocr_outputs[0], chars)
                ocr_texts.append(text)

        rows.append({
            'image': image_path.name,
            'index': idx,
            'class_id': cls_id,
            'class_name': name,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'det_conf': det['score'],
            'ocr_text': text,
        })

        color = CLASS_COLORS.get(cls_id, (0, 220, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f'{name} {det["score"]:.2f}'
        if text:
            label = f'{text} {det["score"]:.2f}'
        cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    t2 = time.time()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out_img = args.save_dir / f'{image_path.stem}_result.jpg'
    cv2.imwrite(str(out_img), vis)

    counts = {}
    for det in dets:
        name = class_name(det['class_id'], classes)
        counts[name] = counts.get(name, 0) + 1
    print(f'input: {image_path}')
    print(f'YOLO detect: {len(dets)} object(s), counts: {counts}, time: {(t1 - t0) * 1000:.1f} ms')
    if args.skip_ocr:
        print('OCR result: <skipped>')
    elif ocr_texts:
        print('OCR result:', ', '.join(ocr_texts))
    else:
        print('OCR result: <no TBJU_region>')
    print(f'OCR+draw time: {(t2 - t1) * 1000:.1f} ms')
    print(f'Saved result to: {out_img}')
    return rows


def collect_images(args):
    if args.image:
        return [Path(args.image)]
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted(p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in suffixes)


def parse_args():
    base = Path(__file__).resolve().parent.parent  # RKNN_deploy/
    p = argparse.ArgumentParser(description='Merged YOLO + TBJU OCR RKNN inference')
    p.add_argument('--image', type=str, default=None)
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--save_dir', type=Path, default=base / 'output' / 'results_merged')
    p.add_argument('--yolo_model', type=Path, default=base / 'models' / 'yolo' / 'merged_yolov8.rknn')
    p.add_argument('--ocr_model', type=Path, default=base / 'models' / 'ocr' / 'rec_tbju.rknn')
    p.add_argument('--dict', type=Path, default=base / 'config' / 'ppocr_keys_v1.txt')
    p.add_argument('--classes', type=Path, default=base / 'config' / 'merged_classes.txt')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--max_det', type=int, default=100)
    p.add_argument('--crop_pad', type=int, default=4)
    p.add_argument('--tbju_class_id', type=int, default=0)
    p.add_argument('--skip_ocr', action='store_true')
    p.add_argument('--all_cores', action='store_true')
    args = p.parse_args()
    if not args.image and not args.image_dir:
        p.error('please provide --image or --image_dir')
    return args


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    print(f'classes: {classes}')
    print(f'YOLO model: {args.yolo_model}')

    chars = []
    ocr = None
    if not args.skip_ocr:
        chars = load_chars(args.dict)
        print(f'chars: {"".join(chars)}')
        print(f'OCR model: {args.ocr_model}')

    yolo = RKNNRunner(args.yolo_model, use_all_cores=args.all_cores)
    if not args.skip_ocr:
        ocr = RKNNRunner(args.ocr_model, use_all_cores=args.all_cores)

    try:
        all_rows = []
        for image_path in collect_images(args):
            all_rows.extend(run_one(image_path, yolo, ocr, chars, classes, args))
        csv_path = args.save_dir / 'result.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image', 'index', 'class_id', 'class_name',
                'x1', 'y1', 'x2', 'y2', 'det_conf', 'ocr_text',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'Saved CSV to: {csv_path}')
    finally:
        yolo.release()
        if ocr is not None:
            ocr.release()


if __name__ == '__main__':
    main()
