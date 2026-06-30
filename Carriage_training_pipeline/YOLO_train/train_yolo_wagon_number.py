"""
YOLO 检测模型训练脚本
用于训练 TBJU 车号区域检测模型，支持 RKNN 导出部署到 RK3588
"""

import os
import sys
import shutil
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO


def _copy_dataset(src_dir: Path, dst_dir: Path, label: str):
    """复制单个数据集目录到目标目录"""
    if not src_dir.exists():
        return
    print(f"复制 {label}: {src_dir}")
    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src_split_dir = src_dir / split / subdir
            dst_split_dir = dst_dir / split / subdir
            if src_split_dir.exists():
                count = 0
                for f in src_split_dir.iterdir():
                    if f.is_file():
                        dst_file = dst_split_dir / f.name
                        if dst_file.exists():
                            print(f"  警告: 文件已存在，跳过: {f.name}")
                        else:
                            shutil.copy2(f, dst_file)
                            count += 1
                print(f"  {split}/{subdir}: {count} 个新文件")


def merge_datasets(output_dir: str, dataset_name: str = "wagon_number_detection"):
    """
    合并平视和侧视数据集
    """
    print("=" * 60)
    print("步骤 1: 合并平视和侧视数据集")
    print("=" * 60)

    eye_level_dir = Path(output_dir) / f"{dataset_name}_平视"
    side_level_dir = Path(output_dir) / f"{dataset_name}_侧视"
    merged_dir = Path(output_dir) / f"{dataset_name}_merged"

    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            (merged_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    _copy_dataset(eye_level_dir, merged_dir, "平视数据")
    _copy_dataset(side_level_dir, merged_dir, "侧视数据")

    # 生成 dataset.yaml
    dataset_yaml = {
        "path": str(merged_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["TBJU_region"]
    }

    yaml_path = merged_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False)

    print(f"\n合并完成: {merged_dir}")
    print(f"数据集配置: {yaml_path}")

    # 统计信息
    train_images = len(list((merged_dir / "train" / "images").iterdir()))
    val_images = len(list((merged_dir / "val" / "images").iterdir()))
    train_labels = len(list((merged_dir / "train" / "labels").iterdir()))
    val_labels = len(list((merged_dir / "val" / "labels").iterdir()))

    print(f"\n合并后统计:")
    print(f"  Train: {train_images} 张图片, {train_labels} 个标签")
    print(f"  Val: {val_images} 张图片, {val_labels} 个标签")

    return str(merged_dir), str(yaml_path)


def train_yolo(
    data_yaml: str,
    output_dir: str,
    model_size: str = "s",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "tbju_detection"
):
    """
    训练 YOLO 检测模型
    """
    print("\n" + "=" * 60)
    print("步骤 2: 训练 YOLO 检测模型")
    print("=" * 60)

    # 选择模型大小
    model_map = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt"
    }

    model_name = model_map.get(model_size, "yolov8s.pt")
    print(f"使用模型: {model_name}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")

    # 加载预训练模型
    model = YOLO(model_name)

    # 训练
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
        verbose=True
    )

    print(f"\n训练完成!")
    print(f"最佳模型保存在: {project}/{name}/weights/best.pt")

    return results


def validate_yolo(
    model_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0"
):
    """
    验证 YOLO 模型
    """
    print("\n" + "=" * 60)
    print("步骤 3: 验证 YOLO 模型")
    print("=" * 60)

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=True
    )

    print(f"\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")

    return results


def export_rknn(
    model_path: str,
    output_dir: str,
    img_size: int = 640,
    half: bool = False
):
    """
    导出 RKNN 格式模型
    用于部署到 RK3588 开发板
    """
    print("\n" + "=" * 60)
    print("步骤 4: 导出 RKNN 格式")
    print("=" * 60)

    model = YOLO(model_path)

    # 导出为 ONNX 格式
    onnx_path = model.export(
        format="onnx",
        imgsz=img_size,
        half=half,
        simplify=True,
        opset=12
    )
    print(f"ONNX 模型导出: {onnx_path}")

    # 导出为 RKNN 格式（需要 rknn-toolkit2）
    try:
        rknn_path = model.export(
            format="rknn",
            imgsz=img_size,
            half=half
        )
        print(f"RKNN 模型导出: {rknn_path}")
    except Exception as e:
        print(f"\nRKNN 导出失败: {e}")
        print("请确保已安装 rknn-toolkit2:")
        print("  pip install rknn-toolkit2")
        print("\n或者手动转换:")
        print("  1. 使用导出的 ONNX 模型")
        print("  2. 使用 rknn-toolkit2 进行转换")
        print(f"  参考: https://github.com/airockchip/rknn-toolkit2")

    return onnx_path


def create_inference_script(model_path: str, output_dir: str):
    """
    创建推理脚本
    """
    print("\n" + "=" * 60)
    print("步骤 5: 创建推理脚本")
    print("=" * 60)

    inference_script = '''"""
YOLO 推理脚本
用于测试训练好的模型
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def detect(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.5,
    save_result: bool = True,
    output_dir: str = "results"
):
    """
    单张图片检测
    """
    # 加载模型
    model = YOLO(model_path)

    # 检测
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save_result,
        project=output_dir,
        name="detection",
        exist_ok=True
    )

    # 打印结果
    for result in results:
        boxes = result.boxes
        print(f"检测到 {len(boxes)} 个车号区域:")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            print(f"  [{i+1}] 类别: {cls}, 置信度: {conf:.2f}, 坐标: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return results


def detect_video(
    model_path: str,
    video_path: str,
    conf_threshold: float = 0.5,
    save_result: bool = True,
    output_dir: str = "results"
):
    """
    视频检测
    """
    model = YOLO(model_path)

    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=save_result,
        project=output_dir,
        name="video_detection",
        exist_ok=True
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 车号检测推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--source", type=str, required=True, help="图片或视频路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    parser.add_argument("--save", action="store_true", default=True, help="保存结果")

    args = parser.parse_args()

    # 判断是图片还是视频
    source_path = Path(args.source)
    if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        detect(args.model, args.source, args.conf, args.save, args.output)
    elif source_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        detect_video(args.model, args.source, args.conf, args.save, args.output)
    else:
        print(f"不支持的文件格式: {source_path.suffix}")
'''

    script_path = Path(output_dir) / "inference.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(inference_script)

    print(f"推理脚本已创建: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description="YOLO 车号检测模型训练")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的配置（推荐）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="转换后的数据集目录 (如果使用 --use_config 则无需指定)")
    parser.add_argument("--model_size", type=str, default="s", choices=["n", "s", "m", "l", "x"],
                        help="模型大小 (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--img_size", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (0=GPU0, cpu=CPU)")
    parser.add_argument("--project", type=str, default=None, help="项目保存目录")
    parser.add_argument("--name", type=str, default="tbju_detection", help="实验名称")
    parser.add_argument("--skip_merge", action="store_true", help="跳过数据集合并（如果已合并）")
    parser.add_argument("--merged_dir", type=str, default=None, help="已合并的数据集目录")
    parser.add_argument("--export_rknn", action="store_true", help="导出 RKNN 格式")
    parser.add_argument("--no_inference", action="store_true", help="不创建推理脚本")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO 车号检测模型训练")
    print("=" * 60)

    # 使用配置文件
    if args.use_config:
        # 导入配置文件
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        # 验证路径
        errors = config.validate_paths()
        if errors:
            print("\n路径验证失败:")
            for error in errors:
                print(f"  - {error}")
            print("\n请检查 config.py 中的路径配置")
            return

        config.print_config()
        dataset_dir = config.OUTPUT_DIR
        project_dir = config.YOLO_OUTPUT_DIR
        print(f"\n使用配置文件中的路径")
    else:
        if args.dataset_dir is None:
            print("错误: 请指定 --dataset_dir 或使用 --use_config")
            parser.print_help()
            return
        dataset_dir = args.dataset_dir
        project_dir = args.project if args.project else "runs/detect"

    # 步骤 1: 合并数据集
    if args.skip_merge and args.merged_dir:
        merged_dir = args.merged_dir
        yaml_path = os.path.join(merged_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print(f"错误: 数据集配置不存在: {yaml_path}")
            return
        print(f"使用已合并的数据集: {merged_dir}")
    else:
        merged_dir, yaml_path = merge_datasets(dataset_dir)

    # 步骤 2: 训练模型
    results = train_yolo(
        data_yaml=yaml_path,
        output_dir=dataset_dir,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        project=project_dir,
        name=args.name
    )

    # 步骤 3: 验证模型
    best_model_path = os.path.join(project_dir, args.name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        validate_yolo(
            model_path=best_model_path,
            data_yaml=yaml_path,
            img_size=args.img_size,
            device=args.device
        )

        # 步骤 4: 导出 RKNN（可选）
        if args.export_rknn:
            export_rknn(
                model_path=best_model_path,
                output_dir=os.path.join(project_dir, args.name),
                img_size=args.img_size
            )

        # 步骤 5: 创建推理脚本（可选）
        if not args.no_inference:
            create_inference_script(best_model_path, os.path.join(project_dir, args.name))

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳模型: {best_model_path}")
    print(f"训练结果: {project_dir}/{args.name}/")


if __name__ == "__main__":
    main()
