"""
摄像头实时推理脚本
使用统一 YOLO 模型 + OCR 模型进行实时检测和车号识别。

用法:
  python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt
  python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt --ocr
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# 统一类别名称（6类）
MERGED_NAMES = {
    0: "TBJU_region",
    1: "carriage_rim_region",
    2: "carriage_rim_debris",
    3: "track_region",
    4: "track_intrusion_debris",
    5: "door_region",
}


def load_ocr_model(ocr_model_path, device="cpu"):
    """加载 OCR 模型。"""
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "OCR_train"))
    from ocr_model import PPoCRRec, load_ppocr_model

    return load_ppocr_model(ocr_model_path, device)


def ocr_recognize(model, image, keys, device="cpu"):
    """对裁剪图片进行 OCR 识别。"""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    # OpenCV BGR -> PIL RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # 预处理（高度32，最小宽度384）
    w, h = pil_image.size
    new_h = 32
    new_w = max(384, min(int(w * new_h / h), 384))
    pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

    img_np = np.array(pil_image, dtype=np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float().to(device)

    # 推理
    with torch.no_grad():
        logits = model(img_tensor)
        log_probs = F.log_softmax(logits, dim=-1)

    # CTC 解码
    from ocr_model import ctc_decode
    result = ctc_decode(log_probs, keys)
    return result[0]


def main():
    parser = argparse.ArgumentParser(description="摄像头实时推理")
    parser.add_argument("--model", type=str, required=True, help="YOLO 模型路径 (.pt)")
    parser.add_argument("--ocr", action="store_true", help="启用 OCR 车号识别")
    parser.add_argument("--ocr_model", type=str, default=None,
                        help="OCR 模型路径 (default: OCR_train/output/ppocr_rec_carriage_number/best_model.pth)")
    parser.add_argument("--source", type=str, default="0",
                        help="视频源：0=默认摄像头，或视频文件路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (default: 0.5)")
    parser.add_argument("--img_size", type=int, default=640, help="输入图片尺寸 (default: 640)")
    parser.add_argument("--save", type=str, default=None, help="保存结果视频路径")

    args = parser.parse_args()

    # 加载 YOLO 模型
    print(f"加载 YOLO 模型: {args.model}")
    yolo_model = YOLO(args.model)

    # 加载 OCR 模型
    ocr_model = None
    ocr_keys = None
    if args.ocr:
        ocr_path = args.ocr_model
        if ocr_path is None:
            ocr_path = str(Path(__file__).parent.parent / "OCR_train" / "output" / "ppocr_rec_carriage_number" / "best_model.pth")
        print(f"加载 OCR 模型: {ocr_path}")
        ocr_model, ocr_keys = load_ocr_model(ocr_path)
        print("OCR 模型加载完成")

    # 打开视频源
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {args.source}")
        return

    print(f"视频源已打开: {args.source}")
    print("按 'q' 退出")

    # 视频保存
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"保存视频到: {args.save}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        results = yolo_model(frame, conf=args.conf, imgsz=args.img_size, verbose=False)

        # 先绘制 YOLO 检测结果
        annotated = results[0].plot()

        # OCR 识别（在 YOLO 绘制之后再画文字）
        if ocr_model is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # TBJU_region
                    # 裁剪车号区域
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        text = ocr_recognize(ocr_model, crop, ocr_keys)
                        # 在框下方显示识别结果（避免与置信度重叠）
                        cv2.putText(annotated, f"OCR: {text}", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示
        cv2.imshow("Carriage Detection", annotated)

        # 保存
        if writer:
            writer.write(annotated)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
