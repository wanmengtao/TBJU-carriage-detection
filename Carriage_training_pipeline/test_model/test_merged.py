"""
统一检测模型测试脚本
测试合并后的 6 类别 YOLO 模型。

统一类别（nc=6）：
  0: TBJU_region
  1: carriage_rim_region
  2: carriage_rim_debris
  3: track_region
  4: track_intrusion_debris
  5: door_region

用法:
  python test_model/test_merged.py --use_config
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO

# 统一类别配置
MERGED_CONFIG = {
    "dataset_name": "merged_detection",
    "display_name": "统一检测",
    "nc": 6,
    "names": [
        "TBJU_region",
        "carriage_rim_region",
        "carriage_rim_debris",
        "track_region",
        "track_intrusion_debris",
        "door_region",
    ],
}


def test_yolo_detection(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    iou: float = 0.5,
    split: str = "test",
):
    """运行 YOLO 模型验证，返回指标。"""
    print("\n" + "=" * 60)
    print("测试 YOLO 统一检测模型")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        iou=iou,
        split=split,
        verbose=True,
    )

    metrics = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "F1": 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-16),
    }

    print(f"\n检测结果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, results


def test_per_class(model_path: str, data_yaml: str, img_size: int, device: str, class_names: list, split: str = "test"):
    """按类别统计检测指标。"""
    print("\n" + "=" * 60)
    print("按类别统计")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, device=device, split=split, verbose=False)

    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(results.box.ap50):
            per_class[name] = {
                "AP50": results.box.ap50[i],
                "AP": results.box.ap[i] if i < len(results.box.ap) else 0,
            }
            print(f"  {name}: AP50={per_class[name]['AP50']:.4f}, AP={per_class[name]['AP']:.4f}")

    return per_class


def save_test_report(output_dir: str, metrics: dict, per_class: dict, model_path: str, data_yaml: str, split: str):
    """保存测试报告。"""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "test_report_merged.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("统一检测模型测试报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"数据集: {data_yaml}\n")
        f.write(f"测试集: {split}\n\n")
        f.write("整体指标:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n按类别指标:\n")
        for name, cls_metrics in per_class.items():
            f.write(f"  {name}:\n")
            for k, v in cls_metrics.items():
                f.write(f"    {k}: {v:.4f}\n")

    print(f"\n测试报告: {report_path}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "test_results_merged.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.4f}"])
        for name, cls_metrics in per_class.items():
            for k, v in cls_metrics.items():
                writer.writerow([f"{name}_{k}", f"{v:.4f}"])

    print(f"测试结果 CSV: {csv_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="统一检测模型测试")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的路径配置（推荐）")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径（不使用 config 时可指定）")
    parser.add_argument("--dataset_yaml", type=str, default=None,
                        help="数据集 YAML 路径（不使用 config 时可指定）")
    parser.add_argument("--img_size", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--device", type=str, default="0", help="计算设备")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="测试集分割 (default: test)")

    args = parser.parse_args()

    print("=" * 60)
    print(f"{MERGED_CONFIG['display_name']}模型测试")
    print(f"类别数: {MERGED_CONFIG['nc']}")
    print(f"类别: {MERGED_CONFIG['names']}")
    print("=" * 60)

    # 解析路径
    if args.use_config:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        model_path = str(config.PROJECT_ROOT / "YOLO_train" / "run_merged" / "train" / "weights" / "best.pt")
        dataset_yaml = str(config.DATASETS_DIR / MERGED_CONFIG["dataset_name"] / "dataset.yaml")
        output_dir = str(config.TEST_RESULTS_DIR / "merged")
    else:
        model_path = args.model_path
        dataset_yaml = args.dataset_yaml
        output_dir = args.output or "test_model/results/merged"

        if not model_path or not dataset_yaml:
            print("错误: 不使用 --use_config 时需指定 --model_path 和 --dataset_yaml")
            parser.print_help()
            return

    # 检查文件
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        return
    if not os.path.exists(dataset_yaml):
        print(f"错误: 数据集配置不存在: {dataset_yaml}")
        return

    print(f"模型: {model_path}")
    print(f"数据集: {dataset_yaml}")
    print(f"测试集: {args.split}")

    # 测试
    metrics, results = test_yolo_detection(
        model_path=model_path,
        data_yaml=dataset_yaml,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        iou=args.iou,
        split=args.split,
    )

    # 按类别统计
    per_class = test_per_class(
        model_path=model_path,
        data_yaml=dataset_yaml,
        img_size=args.img_size,
        device=args.device,
        class_names=MERGED_CONFIG["names"],
        split=args.split,
    )

    # 保存报告
    save_test_report(output_dir, metrics, per_class, model_path, dataset_yaml, args.split)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
