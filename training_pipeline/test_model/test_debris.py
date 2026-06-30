"""
异物检测模型测试脚本
用于测试训练好的异物检测 YOLO 模型（车厢沿异物、轨道异物侵限、车门状态）。

用法:
  python test_model/test_debris.py --task carriage_rim_debris --use_config
  python test_model/test_debris.py --task track_intrusion --use_config
  python test_model/test_debris.py --task door_state --use_config
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO


# 任务配置
TASK_CONFIG = {
    "carriage_rim_debris": {
        "dataset_name": "carriage_rim_debris_detection",
        "display_name": "车厢沿异物检测",
        "nc": 2,
        "names": ["region", "debris"],
    },
    "track_intrusion": {
        "dataset_name": "track_intrusion_detection",
        "display_name": "轨道异物侵限检测",
        "nc": 2,
        "names": ["region", "debris"],
    },
    "door_state": {
        "dataset_name": "door_state_detection",
        "display_name": "车门状态检测",
        "nc": 1,
        "names": ["region"],
    },
}


def test_yolo_detection(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
    iou: float = 0.5,
):
    """运行 YOLO 模型验证，返回指标。"""
    print("\n" + "=" * 60)
    print("测试 YOLO 检测模型")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        iou=iou,
        split='test',
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


def test_per_class(model_path: str, data_yaml: str, img_size: int, device: str, class_names: list):
    """按类别统计检测指标。"""
    print("\n" + "=" * 60)
    print("按类别统计")
    print("=" * 60)

    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=img_size, device=device, split='test', verbose=False)

    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(results.box.ap50):
            per_class[name] = {
                "AP50": results.box.ap50[i],
                "AP": results.box.ap[i] if i < len(results.box.ap) else 0,
            }
            print(f"  {name}: AP50={per_class[name]['AP50']:.4f}, AP={per_class[name]['AP']:.4f}")

    return per_class


def save_test_report(output_dir: str, task_name: str, metrics: dict, per_class: dict, model_path: str, data_yaml: str):
    """保存测试报告。"""
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"test_report_{task_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"{task_name} 异物检测测试报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"数据集: {data_yaml}\n\n")
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
    csv_path = os.path.join(output_dir, f"test_results_{task_name}.csv")
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
    parser = argparse.ArgumentParser(description="异物检测模型测试")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()),
                        help="任务名: carriage_rim_debris / track_intrusion / door_state")
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

    args = parser.parse_args()

    task_cfg = TASK_CONFIG[args.task]
    print("=" * 60)
    print(f"{task_cfg['display_name']}模型测试")
    print("=" * 60)

    # 解析路径
    if args.use_config:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        model_path = str(config.YOLO_OUTPUT_DIR / f"{args.task}_detection" / "weights" / "best.pt")
        dataset_yaml = str(config.DATASETS_DIR / "test_output" / f"{task_cfg['dataset_name']}_test" / "dataset.yaml")
        output_dir = str(config.TEST_RESULTS_DIR / args.task)
    else:
        model_path = args.model_path
        dataset_yaml = args.dataset_yaml
        output_dir = args.output or f"test_model/results/{args.task}"

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

    # 测试
    metrics, results = test_yolo_detection(
        model_path=model_path,
        data_yaml=dataset_yaml,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        iou=args.iou,
    )

    # 按类别统计
    per_class = test_per_class(
        model_path=model_path,
        data_yaml=dataset_yaml,
        img_size=args.img_size,
        device=args.device,
        class_names=task_cfg["names"],
    )

    # 保存报告
    save_test_report(output_dir, args.task, metrics, per_class, model_path, dataset_yaml)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
