"""
YOLO 统一检测模型训练脚本
将车号检测、车厢沿异物、轨道异物侵限、车门状态四个任务合并为一个模型。

统一类别（nc=6）：
  0: TBJU_region
  1: carriage_rim_region
  2: carriage_rim_debris
  3: track_region
  4: track_intrusion_debris
  5: door_region

用法:
  python YOLO_train/train_yolo_merged.py --use_config
"""

import os
import sys
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


def train_yolo(
    data_yaml: str,
    model_size: str = "s",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "merged_detection",
):
    """训练 YOLO 检测模型。"""
    print("\n" + "=" * 60)
    print("训练 YOLO 统一检测模型")
    print("=" * 60)

    model_map = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    }

    model_name = model_map.get(model_size, "yolov8s.pt")
    print(f"使用模型: {model_name}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        verbose=True,
    )

    print(f"\n训练完成!")
    print(f"最佳模型: {project}/{name}/weights/best.pt")

    return results


def validate_yolo(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0",
):
    """验证 YOLO 模型。"""
    print("\n" + "=" * 60)
    print("验证 YOLO 模型")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=True,
    )

    print(f"\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    # 按类别统计
    print(f"\n按类别 AP50:")
    for i, name in enumerate(MERGED_CONFIG["names"]):
        if i < len(results.box.ap50):
            print(f"  {name}: {results.box.ap50[i]:.4f}")

    return results


def export_onnx(model_path: str, img_size: int = 640, half: bool = False):
    """导出 ONNX 格式模型。"""
    print("\n" + "=" * 60)
    print("导出 ONNX 模型")
    print("=" * 60)

    model = YOLO(model_path)

    onnx_path = model.export(format="onnx", imgsz=img_size, half=half, simplify=True, opset=12)
    print(f"ONNX 模型: {onnx_path}")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="YOLO 统一检测模型训练")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的路径配置（推荐）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="数据集目录（不使用 config 时必填）")
    parser.add_argument("--model_size", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"],
                        help="模型大小 (default: s)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小 (default: 16)")
    parser.add_argument("--img_size", type=int, default=640, help="输入图片尺寸 (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (default: 0)")
    parser.add_argument("--no_export", action="store_true", help="跳过 ONNX 导出")

    args = parser.parse_args()

    print("=" * 60)
    print(f"YOLO {MERGED_CONFIG['display_name']}模型训练")
    print(f"类别数: {MERGED_CONFIG['nc']}")
    print(f"类别: {MERGED_CONFIG['names']}")
    print("=" * 60)

    # 解析路径
    if args.use_config:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        dataset_dir = config.DATASETS_DIR / MERGED_CONFIG["dataset_name"]
        project_dir = str(config.PROJECT_ROOT / "YOLO_train" / "run_merged")
    else:
        if args.dataset_dir is None:
            print("错误: 请指定 --dataset_dir 或使用 --use_config")
            parser.print_help()
            return
        dataset_dir = Path(args.dataset_dir)
        project_dir = str(Path(__file__).parent / "run_merged")

    # 检查 dataset.yaml
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        print(f"错误: dataset.yaml 不存在: {yaml_path}")
        return

    print(f"\n数据集: {dataset_dir}")
    print(f"配置文件: {yaml_path}")
    print(f"输出目录: {project_dir}")

    # 训练
    results = train_yolo(
        data_yaml=str(yaml_path),
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=project_dir,
        name="",
    )

    # 验证 + 导出 ONNX
    best_model_path = os.path.join(project_dir, "train", "weights", "best.pt")
    if os.path.exists(best_model_path):
        validate_yolo(
            model_path=best_model_path,
            data_yaml=str(yaml_path),
            img_size=args.img_size,
            device=args.device,
        )

        if not args.no_export:
            onnx_path = export_onnx(model_path=best_model_path, img_size=args.img_size)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳模型 (.pt):  {best_model_path}")
    if not args.no_export:
        print(f"ONNX 模型 (.onnx): {onnx_path}")
    print(f"\nRKNN 转换: 使用 rknn-toolkit2 将 ONNX 转为 RKNN 格式部署到 RK3588")


if __name__ == "__main__":
    main()
