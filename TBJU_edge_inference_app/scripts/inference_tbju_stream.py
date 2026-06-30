#!/usr/bin/env python3
import argparse
import csv
import time
from pathlib import Path

import cv2

from inference_tbju import (
    CLASS_COLORS,
    RKNNRunner,
    class_name,
    ctc_decode,
    decode_yolov8,
    expand_box,
    load_chars,
    load_classes,
    preprocess_ocr,
    preprocess_yolo,
)


def draw_rows(frame, rows, classes, fps_text=''):
    vis = frame.copy()
    for row in rows:
        cls_id = row['class_id']
        name = class_name(cls_id, classes)
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        color = CLASS_COLORS.get(cls_id, (0, 220, 0))
        label = f'{name} {row["det_conf"]:.2f}'
        if row.get('ocr_text'):
            label = f'{row["ocr_text"]} {row["det_conf"]:.2f}'
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if fps_text:
        cv2.rectangle(vis, (8, 8), (420, 42), (0, 0, 0), -1)
        cv2.putText(vis, fps_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis


def infer_frame(frame, yolo, ocr, chars, classes, args, source, frame_id):
    t0 = time.time()
    yolo_input, ratio, pad = preprocess_yolo(frame)
    yolo_outputs = yolo.run(yolo_input)
    dets = decode_yolov8(yolo_outputs, ratio, pad, frame.shape, args.conf, args.iou, args.max_det)
    t1 = time.time()

    rows = []
    ocr_texts = []
    for idx, det in enumerate(dets):
        cls_id = det['class_id']
        name = class_name(cls_id, classes)
        box = expand_box(det['bbox'], args.crop_pad if cls_id == args.tbju_class_id else 0, frame.shape)
        x1, y1, x2, y2 = box
        text = ''
        if (not args.skip_ocr) and cls_id == args.tbju_class_id and ocr is not None:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                ocr_input = preprocess_ocr(crop)
                ocr_outputs = ocr.run(ocr_input, data_format='nchw')
                text = ctc_decode(ocr_outputs[0], chars)
                ocr_texts.append(text)
        rows.append({
            'source': source,
            'frame': frame_id,
            'index': idx,
            'class_id': cls_id,
            'class_name': name,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'det_conf': float(det['score']),
            'ocr_text': text,
        })

    t2 = time.time()
    return rows, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, ocr_texts


def make_writer(output_path, fps, width, height):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if suffix == '.mp4' else 'MJPG'))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened() and suffix == '.mp4':
        fallback = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(fallback), fourcc, fps, (width, height))
        output_path = fallback
    if not writer.isOpened():
        raise RuntimeError(f'failed to open VideoWriter: {output_path}')
    return writer, output_path


def open_camera(args):
    if args.gst:
        fps_num = max(1, int(round(args.fps)))
        pipeline = (
            f'v4l2src device={args.camera} io-mode=2 ! '
            f'video/x-raw,format=NV12,width={args.width},height={args.height},framerate={fps_num}/1 ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink drop=true max-buffers=1 sync=false'
        )
        print(f'GStreamer camera pipeline: {pipeline}')
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def run_stream(cap, yolo, ocr, chars, classes, args, source_name, default_output):
    if not cap.isOpened():
        raise RuntimeError(
            f'failed to open video source: {source_name}. '
            'If --gst was used, test the same camera with gst-launch first, '
            'or try without --gst.'
        )

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError(f'failed to read first frame from {source_name}')

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = args.fps

    output = args.output or default_output
    writer = None
    output_path = None
    if output:
        writer, output_path = make_writer(output, fps, width, height)
        print(f'output video: {output_path}')

    if args.csv:
        csv_path = Path(args.csv)
    elif output_path or default_output:
        csv_path = Path(output_path or default_output).with_suffix('.csv')
    else:
        csv_path = Path('results_camera') / 'camera_result.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer_csv = csv.DictWriter(csv_file, fieldnames=[
        'source', 'frame', 'index', 'class_id', 'class_name',
        'x1', 'y1', 'x2', 'y2', 'det_conf', 'ocr_text',
    ])
    writer_csv.writeheader()

    if args.show:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    frame_id = 0
    processed = 0
    last_rows = []
    start = time.time()

    try:
        while True:
            if frame_id % args.every_n == 0:
                rows, yolo_ms, ocr_ms, ocr_texts = infer_frame(
                    frame, yolo, ocr, chars, classes, args, source_name, frame_id
                )
                last_rows = rows
                for row in rows:
                    writer_csv.writerow(row)
                processed += 1
                fps_now = processed / max(time.time() - start, 1e-6)
                fps_text = f'FPS {fps_now:.1f} | YOLO {yolo_ms:.1f} ms | OCR {ocr_ms:.1f} ms | frame {frame_id}'
                if ocr_texts:
                    print(f'frame {frame_id}: OCR={",".join(ocr_texts)}; det={len(rows)}; {yolo_ms:.1f}+{ocr_ms:.1f} ms')
                else:
                    print(f'frame {frame_id}: det={len(rows)}; {yolo_ms:.1f}+{ocr_ms:.1f} ms')
            else:
                fps_text = f'skip infer | every_n={args.every_n} | frame {frame_id}'

            vis = draw_rows(frame, last_rows, classes, fps_text)
            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow(args.window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break

            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        csv_file.close()
        if writer is not None:
            writer.release()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f'saved csv: {csv_path}')
    if output_path:
        print(f'saved video: {output_path}')


def parse_args():
    base = Path(__file__).resolve().parent.parent  # RKNN_deploy/
    p = argparse.ArgumentParser(description='Merged YOLO + TBJU OCR RKNN video/camera demo')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=str, default=None)
    src.add_argument('--camera', type=str, default=None)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--csv', type=str, default=None)
    p.add_argument('--show', action='store_true')
    p.add_argument('--gst', action='store_true', help='use GStreamer pipeline for MIPI camera')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--every_n', type=int, default=1)
    p.add_argument('--max_frames', type=int, default=0)
    p.add_argument('--window_name', type=str, default='Merged RKNN Demo')
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
    args.every_n = max(1, args.every_n)
    return args


def main():
    args = parse_args()
    classes = load_classes(args.classes)
    chars = []
    ocr = None
    print(f'classes: {classes}')
    print(f'YOLO model: {args.yolo_model}')
    yolo = RKNNRunner(args.yolo_model, use_all_cores=args.all_cores)
    if not args.skip_ocr:
        chars = load_chars(args.dict)
        print(f'OCR model: {args.ocr_model}')
        print(f'chars: {"".join(chars)}')
        ocr = RKNNRunner(args.ocr_model, use_all_cores=args.all_cores)

    try:
        if args.video:
            video_path = Path(args.video)
            cap = cv2.VideoCapture(str(video_path))
            default_output = str(Path('results_video') / f'{video_path.stem}_result.avi')
            run_stream(cap, yolo, ocr, chars, classes, args, video_path.name, default_output)
        else:
            cap = open_camera(args)
            default_output = str(Path('results_camera') / 'camera_result.avi') if args.output else None
            run_stream(cap, yolo, ocr, chars, classes, args, str(args.camera), default_output)
    finally:
        yolo.release()
        if ocr is not None:
            ocr.release()


if __name__ == '__main__':
    main()
