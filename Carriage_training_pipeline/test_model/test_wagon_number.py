"""
模型测试脚本
用于测试训练好的 YOLO 检测模型和 OCR 识别模型
使用转换后的测试数据集
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置文件导入
# ============================================================

def get_config():
    """获取配置文件路径"""
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    try:
        import config
        return config
    except ImportError:
        return None


# ============================================================
# 从 ocr_model.py 导入共享模型定义
# ============================================================

def _import_ocr_model():
    """导入 OCR 模型共享模块"""
    parent_dir = Path(__file__).parent.parent
    ocr_train_dir = parent_dir / "OCR_train"
    if str(ocr_train_dir) not in sys.path:
        sys.path.insert(0, str(ocr_train_dir))
    from ocr_model import (
        PPoCRRec, ctc_decode, load_ppocr_model,
        preprocess_ocr_image, ocr_recognize,
    )
    return PPoCRRec, ctc_decode, load_ppocr_model, preprocess_ocr_image, ocr_recognize


# 延迟导入，避免在不需要 OCR 测试时也导入
_ocr_module = None


def _get_ocr_module():
    global _ocr_module
    if _ocr_module is None:
        _ocr_module = _import_ocr_model()
    return _ocr_module


def load_ocr_model(model_path: str, device: str = "cuda"):
    """加载 PP-OCR Rec 模型（包装函数）"""
    _, _, load_ppocr_model_fn, _, _ = _get_ocr_module()
    return load_ppocr_model_fn(model_path, device)


def recognize_text(model, chars, image: Image.Image, device: str = "cuda") -> str:
    """单张图片 OCR 识别（包装函数）"""
    _, _, _, _, ocr_recognize_fn = _get_ocr_module()
    return ocr_recognize_fn(model, chars, image, device)


def map_device_for_ocr(device: str) -> str:
    """
    将 YOLO 格式的 device 参数映射为 OCR 模型可用的格式
    YOLO: "0", "1", "cpu"
    OCR (PyTorch): "cuda", "cuda:0", "cpu"
    """
    if device == "cpu":
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


# ============================================================
# 数据加载
# ============================================================

def load_yolo_labels(label_path: str, img_width: int, img_height: int) -> List[Dict]:
    """加载 YOLO 格式标签文件"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append({
                    "class_id": class_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "x_center": x_center, "y_center": y_center,
                    "width": width, "height": height
                })

    return boxes


def load_ocr_labels(csv_path: str) -> List[Dict]:
    """加载 OCR 格式标签文件 (labels.csv)"""
    labels = []
    if not os.path.exists(csv_path):
        return labels

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                "crop_name": row["crop_name"],
                "original_image": row["original_image"],
                "text": row["text"],
                "type": row.get("type", "TBJU_region"),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "split": row.get("split", "test"),
                "view_type": row.get("view_type", "unknown")
            })

    return labels


def load_test_dataset(dataset_dir: str) -> Dict:
    """加载转换后的测试数据集"""
    dataset_path = Path(dataset_dir)

    mapping_path = dataset_path / "test_data_mapping.json"
    mapping = {}
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    # YOLO 数据
    yolo_dir = dataset_path / "wagon_number_detection_test"
    yolo_images = {}
    yolo_labels = {}

    if yolo_dir.exists():
        images_dir = yolo_dir / "test" / "images"
        labels_dir = yolo_dir / "test" / "labels"

        if images_dir.exists():
            for img_file in images_dir.iterdir():
                if img_file.is_file():
                    img = Image.open(img_file)
                    yolo_images[img_file.name] = {
                        "path": str(img_file),
                        "width": img.width,
                        "height": img.height
                    }

        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                img_name = label_file.stem + ".jpg"
                if img_name in yolo_images:
                    img_info = yolo_images[img_name]
                    boxes = load_yolo_labels(
                        str(label_file),
                        img_info["width"],
                        img_info["height"]
                    )
                    yolo_labels[img_name] = boxes

    # OCR 数据
    ocr_dir = dataset_path / "wagon_number_ocr_test"
    ocr_crops = {}
    ocr_labels = []

    if ocr_dir.exists():
        crops_dir = ocr_dir / "test" / "crops"
        csv_path = ocr_dir / "test" / "labels.csv"

        if crops_dir.exists():
            for crop_file in crops_dir.iterdir():
                if crop_file.is_file():
                    ocr_crops[crop_file.name] = str(crop_file)

        if csv_path.exists():
            ocr_labels = load_ocr_labels(str(csv_path))

    dataset = {
        "mapping": mapping,
        "yolo": {
            "images": yolo_images,
            "labels": yolo_labels,
            "yaml_path": str(yolo_dir / "dataset.yaml") if yolo_dir.exists() else None
        },
        "ocr": {
            "crops": ocr_crops,
            "labels": ocr_labels
        }
    }

    print(f"加载测试数据集:")
    print(f"  YOLO: {len(yolo_images)} 张图片, {len(yolo_labels)} 个标签文件")
    print(f"  OCR: {len(ocr_crops)} 个 crop, {len(ocr_labels)} 条标签")

    return dataset


# ============================================================
# YOLO 检测测试
# ============================================================

def test_yolo_detection(
    dataset: Dict,
    model_path: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = "0"
) -> Dict:
    """测试 YOLO 检测模型"""
    print("\n" + "=" * 60)
    print("测试 YOLO 检测模型")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 请安装 ultralytics: pip install ultralytics")
        return {}

    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        return {}

    yaml_path = dataset["yolo"]["yaml_path"]
    if not yaml_path or not os.path.exists(yaml_path):
        print(f"错误: 数据集配置不存在")
        return {}

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"\n运行验证...")
    results = model.val(
        data=yaml_path,
        imgsz=640,
        batch=16,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=True
    )

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "f1_score": 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0,
        "total_images": len(dataset["yolo"]["images"]),
        "total_labels": sum(len(v) for v in dataset["yolo"]["labels"].values())
    }

    print(f"\nYOLO 检测结果:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")

    return metrics


# ============================================================
# OCR 识别测试
# ============================================================

def test_ocr_recognition(
    dataset: Dict,
    model_path: str,
    device: str = "cuda"
) -> Dict:
    """测试 OCR 识别模型 (PP-OCR Rec)"""
    print("\n" + "=" * 60)
    print("测试 OCR 识别模型")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        return {}

    print(f"加载模型: {model_path}")
    model, chars = load_ocr_model(model_path, device)

    ocr_labels = dataset["ocr"]["labels"]
    ocr_crops = dataset["ocr"]["crops"]

    if not ocr_labels:
        print("错误: 没有测试数据")
        return {}

    print(f"\n测试 {len(ocr_labels)} 个样本...")

    all_results = []
    correct = 0
    total = 0
    by_view = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, label in enumerate(ocr_labels):
        crop_name = label["crop_name"]
        ground_truth = label["text"]
        view_type = label.get("view_type", "unknown")

        crop_path = ocr_crops.get(crop_name)
        if not crop_path or not os.path.exists(crop_path):
            continue

        image = Image.open(crop_path).convert("RGB")
        predicted_text = recognize_text(model, chars, image, device)

        is_correct = predicted_text.lower() == ground_truth.lower()
        if is_correct:
            correct += 1
        total += 1

        by_view[view_type]["total"] += 1
        if is_correct:
            by_view[view_type]["correct"] += 1

        all_results.append({
            "crop_name": crop_name,
            "original_image": label["original_image"],
            "view_type": view_type,
            "ground_truth": ground_truth,
            "predicted": predicted_text,
            "correct": is_correct
        })

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i + 1}/{len(ocr_labels)} 个样本")

    accuracy = correct / total if total > 0 else 0

    view_metrics = {}
    for view_type, stats in by_view.items():
        view_metrics[view_type] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        }

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_view": view_metrics,
        "details": all_results
    }

    print(f"\nOCR 识别结果:")
    print(f"  总样本数: {total}")
    print(f"  正确数: {correct}")
    print(f"  准确率: {accuracy:.4f}")
    for view_type, vm in view_metrics.items():
        print(f"  {view_type}: {vm['accuracy']:.4f} ({vm['correct']}/{vm['total']})")

    return metrics


# ============================================================
# 端到端测试
# ============================================================

def test_end_to_end(
    dataset: Dict,
    yolo_model_path: str,
    ocr_model_path: str,
    conf_threshold: float = 0.5,
    device: str = "0"
) -> Dict:
    """端到端测试: YOLO 检测 + OCR 识别"""
    print("\n" + "=" * 60)
    print("端到端测试: YOLO + OCR")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 请安装 ultralytics")
        return {}

    print(f"加载 YOLO 模型: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    # OCR 使用 PyTorch 格式的 device
    ocr_device = map_device_for_ocr(device)
    print(f"加载 OCR 模型: {ocr_model_path}")
    ocr_model, ocr_chars = load_ocr_model(ocr_model_path, ocr_device)

    yolo_images = dataset["yolo"]["images"]
    yolo_labels = dataset["yolo"]["labels"]

    if not yolo_images:
        print("错误: 没有测试数据")
        return {}

    print(f"\n测试 {len(yolo_images)} 张图片...")

    total_gt = 0
    total_detected = 0
    correct_e2e = 0
    all_results = []

    for i, (img_name, img_info) in enumerate(yolo_images.items()):
        image_path = img_info["path"]

        gt_texts = []

        for ocr_label in dataset["ocr"]["labels"]:
            if ocr_label["original_image"] == img_name:
                gt_texts.append(ocr_label["text"])

        total_gt += len(gt_texts)

        # YOLO 检测
        detections = yolo_model.predict(
            source=image_path,
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        det_boxes = []
        for det in detections:
            for box in det.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                det_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf
                })

        total_detected += len(det_boxes)

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # 对每个检测框进行 OCR
        detected_texts = []
        for det_box in det_boxes:
            x1 = max(0, int(det_box["x1"]))
            y1 = max(0, int(det_box["y1"]))
            x2 = min(img_w, int(det_box["x2"]))
            y2 = min(img_h, int(det_box["y2"]))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            text = recognize_text(ocr_model, ocr_chars, crop, ocr_device)
            detected_texts.append(text)

        # 计算匹配
        matched = 0
        for gt_text in gt_texts:
            if gt_text in detected_texts:
                matched += 1
                correct_e2e += 1

        all_results.append({
            "image_name": img_name,
            "gt_count": len(gt_texts),
            "detected_count": len(detected_texts),
            "matched": matched,
            "gt_texts": gt_texts,
            "detected_texts": detected_texts
        })

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(yolo_images)} 张图片")

    e2e_accuracy = correct_e2e / total_gt if total_gt > 0 else 0

    metrics = {
        "total_gt": total_gt,
        "total_detected": total_detected,
        "correct_e2e": correct_e2e,
        "e2e_accuracy": e2e_accuracy,
        "details": all_results
    }

    print(f"\n端到端测试结果:")
    print(f"  总标注框数: {total_gt}")
    print(f"  检测框数: {total_detected}")
    print(f"  完全匹配: {correct_e2e}")
    print(f"  端到端准确率: {e2e_accuracy:.4f}")

    return metrics


# ============================================================
# 结果保存
# ============================================================

def save_test_results(
    yolo_metrics: Dict,
    ocr_metrics: Dict,
    e2e_metrics: Dict,
    output_dir: str,
    dataset: Dict
):
    """保存测试结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 保存汇总报告
    report_path = output_path / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("模型测试报告\n")
        f.write("=" * 60 + "\n\n")

        mapping = dataset.get("mapping", {})
        f.write("测试数据统计:\n")
        f.write(f"  总样本数: {mapping.get('total_samples', 'N/A')}\n")
        f.write(f"  平视: {mapping.get('eye_level_count', 'N/A')} 个\n")
        f.write(f"  侧视: {mapping.get('side_level_count', 'N/A')} 个\n")
        f.write(f"  需复核: {mapping.get('review_count', 'N/A')} 个\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("YOLO 检测结果:\n")
        if yolo_metrics:
            f.write(f"  mAP50: {yolo_metrics.get('mAP50', 0):.4f}\n")
            f.write(f"  mAP50-95: {yolo_metrics.get('mAP50-95', 0):.4f}\n")
            f.write(f"  Precision: {yolo_metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall: {yolo_metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1 Score: {yolo_metrics.get('f1_score', 0):.4f}\n")
        else:
            f.write("  未测试\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("OCR 识别结果:\n")
        if ocr_metrics:
            f.write(f"  总样本数: {ocr_metrics.get('total', 0)}\n")
            f.write(f"  正确数: {ocr_metrics.get('correct', 0)}\n")
            f.write(f"  准确率: {ocr_metrics.get('accuracy', 0):.4f}\n")
            for vt, vm in ocr_metrics.get('by_view', {}).items():
                f.write(f"  {vt}: {vm['accuracy']:.4f} ({vm['correct']}/{vm['total']})\n")
        else:
            f.write("  未测试\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("端到端测试结果:\n")
        if e2e_metrics:
            f.write(f"  总标注框数: {e2e_metrics.get('total_gt', 0)}\n")
            f.write(f"  检测框数: {e2e_metrics.get('total_detected', 0)}\n")
            f.write(f"  完全匹配: {e2e_metrics.get('correct_e2e', 0)}\n")
            f.write(f"  端到端准确率: {e2e_metrics.get('e2e_accuracy', 0):.4f}\n")
        else:
            f.write("  未测试\n")

    print(f"\n汇总报告: {report_path}")

    # 2. 保存 OCR 详细结果 CSV
    if ocr_metrics and "details" in ocr_metrics:
        ocr_csv = output_path / "ocr_recognition_details.csv"
        with open(ocr_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "crop_name", "original_image", "view_type",
                "ground_truth", "predicted", "correct"
            ])
            writer.writeheader()
            for detail in ocr_metrics["details"]:
                writer.writerow(detail)
        print(f"OCR 详细结果: {ocr_csv}")

    # 3. 保存错误样本列表
    if ocr_metrics and "details" in ocr_metrics:
        errors_csv = output_path / "ocr_errors.csv"
        errors = [d for d in ocr_metrics["details"] if not d.get("correct", True)]
        if errors:
            with open(errors_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "crop_name", "original_image", "view_type",
                    "ground_truth", "predicted"
                ])
                writer.writeheader()
                for err in errors:
                    writer.writerow(err)
            print(f"OCR 错误样本: {errors_csv} ({len(errors)} 个)")

    # 4. 保存端到端详细结果
    if e2e_metrics and "details" in e2e_metrics:
        e2e_csv = output_path / "end_to_end_details.csv"
        with open(e2e_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_name", "gt_count", "detected_count",
                "matched", "gt_texts", "detected_texts"
            ])
            writer.writeheader()
            for detail in e2e_metrics["details"]:
                writer.writerow({
                    "image_name": detail.get("image_name", ""),
                    "gt_count": detail.get("gt_count", 0),
                    "detected_count": detail.get("detected_count", 0),
                    "matched": detail.get("matched", 0),
                    "gt_texts": "|".join(detail.get("gt_texts", [])),
                    "detected_texts": "|".join(detail.get("detected_texts", []))
                })
        print(f"端到端详细结果: {e2e_csv}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="模型测试脚本")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的配置（推荐）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="测试数据集目录 (转换后的)")
    parser.add_argument("--yolo_model", type=str, default=None,
                        help="YOLO 检测模型路径")
    parser.add_argument("--ocr_model", type=str, default=None,
                        help="OCR 识别模型路径")
    parser.add_argument("--test_yolo", action="store_true",
                        help="测试 YOLO 检测模型")
    parser.add_argument("--test_ocr", action="store_true",
                        help="测试 OCR 识别模型")
    parser.add_argument("--test_e2e", action="store_true",
                        help="测试端到端 (YOLO + OCR)")
    parser.add_argument("--test_all", action="store_true",
                        help="测试所有")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO 置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU 匹配阈值")
    parser.add_argument("--device", type=str, default="0",
                        help="计算设备 (0=GPU0, cpu=CPU)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("模型测试脚本")
    print("=" * 60)

    # 使用配置文件
    if args.use_config:
        config = get_config()
        if config is None:
            print("错误: 无法加载 config.py")
            return

        errors = config.validate_paths()
        if errors:
            print("\n路径验证失败:")
            for error in errors:
                print(f"  - {error}")
            return

        config.print_config()

        dataset_dir = args.dataset_dir or config.TEST_OUTPUT_DIR
        output_dir = args.output or str(config.TEST_RESULTS_DIR / "wagon_number")

        if args.yolo_model is None:
            args.yolo_model = os.path.join(
                config.YOLO_OUTPUT_DIR, "tbju_detection", "weights", "best.pt"
            )
        if args.ocr_model is None:
            args.ocr_model = os.path.join(
                config.OCR_MODEL_DIR, "best_model.pth"
            )
    else:
        if args.dataset_dir is None:
            print("错误: 请指定 --dataset_dir 或使用 --use_config")
            parser.print_help()
            return

        dataset_dir = args.dataset_dir
        output_dir = args.output or "test_model/results/wagon_number"

    if not (args.test_yolo or args.test_ocr or args.test_e2e or args.test_all):
        args.test_all = True

    print(f"\n加载测试数据集: {dataset_dir}")
    dataset = load_test_dataset(dataset_dir)

    yolo_metrics = {}
    ocr_metrics = {}
    e2e_metrics = {}

    if args.test_all or args.test_yolo:
        yolo_metrics = test_yolo_detection(
            dataset, args.yolo_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )

    if args.test_all or args.test_ocr:
        # OCR 使用 PyTorch 格式的 device
        ocr_device = map_device_for_ocr(args.device)
        ocr_metrics = test_ocr_recognition(
            dataset, args.ocr_model,
            device=ocr_device
        )

    if args.test_all or args.test_e2e:
        e2e_metrics = test_end_to_end(
            dataset, args.yolo_model, args.ocr_model,
            conf_threshold=args.conf,
            device=args.device
        )

    save_test_results(yolo_metrics, ocr_metrics, e2e_metrics, output_dir, dataset)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
