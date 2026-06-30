"""
测试数据转换脚本
将测试集的 Label Studio JSON 标注转换为 YOLO 和 OCR 格式
合并平视和侧视数据，打乱顺序
"""

import os
import sys
import re
import csv
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont


def load_labelstudio_json(json_path: str) -> List[Dict]:
    """
    加载 Label Studio JSON 标注文件
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = []
    for task in data:
        task_id = task.get("id")
        file_upload = task.get("file_upload", "")
        image_path = task.get("data", {}).get("image", "")

        # 获取标注结果
        annotations = task.get("annotations", [])
        results = annotations[0].get("result", []) if annotations else []

        # 配对框和文本
        boxes = {}
        texts = {}

        for result in results:
            result_id = result.get("id")
            result_type = result.get("type")

            if result_type == "rectanglelabels":
                value = result.get("value", {})
                boxes[result_id] = {
                    "x": value.get("x", 0),
                    "y": value.get("y", 0),
                    "width": value.get("width", 0),
                    "height": value.get("height", 0),
                    "label": value.get("rectanglelabels", ["unknown"])[0]
                }
            elif result_type == "textarea":
                value = result.get("value", {})
                text_list = value.get("text", [])
                if text_list:
                    texts[result_id] = text_list[0]

        # 配对
        for box_id, box_info in boxes.items():
            text = texts.get(box_id, "")
            annotations.append({
                "task_id": task_id,
                "file_upload": file_upload,
                "image_path": image_path,
                "box": box_info,
                "text": text
            })

    return annotations


def find_local_image(file_upload: str, images_dir: Path) -> str:
    """
    鲁棒的图片匹配策略
    """
    image_files = {f.name: f for f in images_dir.iterdir() if f.is_file()}

    # 1. 精确匹配
    if file_upload in image_files:
        return str(image_files[file_upload])

    # 2. 去掉 UUID 前缀匹配
    if "-" in file_upload:
        clean_name = "-".join(file_upload.split("-")[1:])
        for fname, fpath in image_files.items():
            if clean_name in fname:
                return str(fpath)

    # 3. 模糊匹配
    for fname, fpath in image_files.items():
        if file_upload.split(".")[0] in fname:
            return str(fpath)

    return None


def normalize_text(text: str) -> str:
    """
    标准化文本：去空格、去横杠、去方括号、转大写
    """
    return text.replace(" ", "").replace("-", "").replace("[", "").replace("]", "").upper()


def validate_text(text: str) -> Tuple[bool, str]:
    """
    验证文本格式
    返回: (is_valid, reason)
    """
    if not text:
        return False, "missing_text"

    # 检查前缀
    if not text.startswith("TBJU") and not text.startswith("TBCU"):
        return False, "invalid_prefix"

    # 检查格式
    if not re.match(r"^(TBJU|TBCU)\d{7}$", text):
        return False, "invalid_format"

    # 检查不确定字符
    if "?" in text:
        return False, "uncertain_chars"

    return True, "valid"


def load_test_dataset(test_dir: str, view_type: str) -> Tuple[List[Dict], List[Dict]]:
    """
    加载测试数据集
    返回: (valid_data, review_data)
    """
    test_path = Path(test_dir)

    # 查找 JSON 标签文件
    labels_dir = test_path / "labels"
    json_files = list(labels_dir.glob("*.json"))

    if not json_files:
        print(f"警告: {labels_dir} 中没有找到 JSON 文件")
        return [], []

    # 加载所有标注
    all_annotations = []
    for json_file in json_files:
        annotations = load_labelstudio_json(str(json_file))
        all_annotations.extend(annotations)

    # 匹配本地图片
    images_dir = test_path / "images"

    # 构建测试数据
    valid_data = []
    review_data = []

    for ann in all_annotations:
        # 查找图片
        image_path = find_local_image(ann["file_upload"], images_dir)

        if not image_path:
            review_data.append({
                "file_upload": ann["file_upload"],
                "text": ann["text"],
                "reason": "image_not_found",
                "view_type": view_type
            })
            continue

        # 标准化文本
        text = normalize_text(ann["text"])

        # 验证文本
        is_valid, reason = validate_text(text)

        item = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "file_upload": ann["file_upload"],
            "view_type": view_type,
            "box": ann["box"],
            "text": ann["text"],
            "text_clean": text,
            "label": ann["box"]["label"],
            "is_valid": is_valid
        }

        if is_valid:
            valid_data.append(item)
        else:
            item["reason"] = reason
            review_data.append(item)

    print(f"加载 {view_type} 测试数据: {len(valid_data)} 有效, {len(review_data)} 需复核")
    return valid_data, review_data


def merge_and_shuffle(
    eye_data: List[Dict],
    side_data: List[Dict],
    seed: int = 42
) -> Tuple[List[Dict], List[int]]:
    """
    合并并打乱数据
    """
    all_data = eye_data + side_data

    random.seed(seed)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    shuffled_data = [all_data[i] for i in indices]

    print(f"\n合并后总数: {len(shuffled_data)} 个标注框")
    print(f"  - 平视: {len(eye_data)} 个")
    print(f"  - 侧视: {len(side_data)} 个")
    print(f"已打乱顺序 (seed={seed})")

    return shuffled_data, indices


def convert_to_yolo_format(
    test_data: List[Dict],
    output_dir: str,
    split: str = "test"
):
    """
    转换为 YOLO 检测数据集格式
    """
    print("\n" + "=" * 60)
    print("转换为 YOLO 检测数据集格式")
    print("=" * 60)

    output_path = Path(output_dir) / f"wagon_number_detection_test"
    images_dir = output_path / split / "images"
    labels_dir = output_path / split / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 按图片分组
    image_groups = defaultdict(list)
    for item in test_data:
        image_groups[item["image_path"]].append(item)

    converted = 0
    for image_path, items in image_groups.items():
        image_path = Path(image_path)

        # 复制图片
        dst_image = images_dir / image_path.name
        if not dst_image.exists():
            shutil.copy2(image_path, dst_image)

        # 生成 YOLO 标签
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        label_file = labels_dir / f"{image_path.stem}.txt"
        with open(label_file, "w") as f:
            for item in items:
                box = item["box"]

                # 转换为 YOLO 格式 (归一化)
                x_center = (box["x"] + box["width"] / 2) / 100
                y_center = (box["y"] + box["height"] / 2) / 100
                width = box["width"] / 100
                height = box["height"] / 100

                # class_id = 0 (TBJU_region)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        converted += 1

    # 生成 dataset.yaml
    yaml_content = f"""path: {output_path}
train: {split}/images
val: {split}/images
nc: 1
names: ['TBJU_region']
"""
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"转换完成: {converted} 张图片")
    print(f"  图片目录: {images_dir}")
    print(f"  标签目录: {labels_dir}")
    print(f"  数据集配置: {yaml_path}")

    return str(output_path), str(yaml_path)


def convert_to_ocr_format(
    test_data: List[Dict],
    output_dir: str,
    split: str = "test"
):
    """
    转换为 OCR 识别数据集格式
    """
    print("\n" + "=" * 60)
    print("转换为 OCR 识别数据集格式")
    print("=" * 60)

    output_path = Path(output_dir) / "wagon_number_ocr_test"
    crops_dir = output_path / split / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 裁剪图片并生成 labels.csv
    labels = []
    preview_dir = output_path / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # 按图片分组用于预览图
    image_groups = defaultdict(list)
    for item in test_data:
        image_groups[item["image_path"]].append(item)

    crop_count = 0
    for i, item in enumerate(test_data):
        image_path = Path(item["image_path"])

        # 加载图片
        with Image.open(image_path) as img:
            img_w, img_h = img.size

            # 裁剪车号区域
            box = item["box"]
            x1 = int(box["x"] / 100 * img_w)
            y1 = int(box["y"] / 100 * img_h)
            x2 = int(x1 + box["width"] / 100 * img_w)
            y2 = int(y1 + box["height"] / 100 * img_h)

            # 确保坐标有效
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪并保存
            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{image_path.stem}_box{i}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, "JPEG", quality=95)

        # 记录标签
        labels.append({
            "crop_name": crop_name,
            "original_image": image_path.name,
            "text": item["text_clean"],
            "type": "TBJU_region",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "split": split,
            "view_type": item["view_type"]
        })

        crop_count += 1

    # 保存 labels.csv
    csv_path = output_path / split / "labels.csv"
    csv_columns = ["crop_name", "original_image", "text", "type", "x1", "y1", "x2", "y2", "split", "view_type"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(labels)

    # 生成预览图
    print(f"\n生成预览图...")
    for image_path, items in image_groups.items():
        generate_preview(image_path, items, preview_dir)

    print(f"\n转换完成: {crop_count} 个 crop")
    print(f"  Crops 目录: {crops_dir}")
    print(f"  标签文件: {csv_path}")
    print(f"  预览目录: {preview_dir}")

    return str(output_path), str(csv_path)


def generate_preview(image_path: str, items: List[Dict], preview_dir: Path):
    """
    生成预览图
    """
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        for item in items:
            box = item["box"]
            x1 = int(box["x"] / 100 * img_w)
            y1 = int(box["y"] / 100 * img_h)
            x2 = int(x1 + box["width"] / 100 * img_w)
            y2 = int(y1 + box["height"] / 100 * img_h)

            # 绿色框表示有效文本
            color = "green" if item["is_valid"] else "red"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 标注文本
            text = item["text_clean"] if item["is_valid"] else item.get("reason", "invalid")
            draw.text((x1, y1 - 15), text, fill=color)

        # 保存预览图
        preview_path = preview_dir / Path(image_path).name
        img.save(preview_path, "JPEG", quality=95)


def save_review_list(review_data: List[Dict], output_dir: str):
    """
    保存复核列表
    """
    if not review_data:
        return

    review_path = Path(output_dir) / "review_list.csv"
    csv_columns = ["file_upload", "text", "reason", "view_type"]

    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(review_data)

    print(f"\n复核列表: {review_path} ({len(review_data)} 条)")


def main():
    parser = argparse.ArgumentParser(description="测试数据转换脚本")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的配置")

    parser.add_argument("--eye_level_dir", type=str, default=None,
                        help="平视测试目录")
    parser.add_argument("--side_level_dir", type=str, default=None,
                        help="侧视测试目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")

    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    print("=" * 60)
    print("测试数据转换")
    print("=" * 60)

    # 使用配置文件
    if args.use_config:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        errors = config.validate_paths()
        if errors:
            print("\n路径验证失败:")
            for error in errors:
                print(f"  - {error}")
            return

        config.print_config()

        eye_level_dir = config.TEST_EYE_LEVEL_DIR
        side_level_dir = config.TEST_SIDE_LEVEL_DIR
        output_dir = args.output or config.TEST_OUTPUT_DIR
    else:
        if args.eye_level_dir is None or args.side_level_dir is None:
            print("错误: 请指定路径或使用 --use_config")
            parser.print_help()
            return

        eye_level_dir = args.eye_level_dir
        side_level_dir = args.side_level_dir
        output_dir = args.output or "test_dataset"

    # 加载数据
    print(f"\n加载测试数据...")
    eye_valid, eye_review = load_test_dataset(eye_level_dir, "平视")
    side_valid, side_review = load_test_dataset(side_level_dir, "侧视")

    # 合并复核列表
    all_review = eye_review + side_review

    # 合并并打乱有效数据
    test_data, shuffle_indices = merge_and_shuffle(eye_valid, side_valid, seed=args.seed)

    if not test_data:
        print("错误: 没有有效数据")
        return

    # 转换为 YOLO 格式
    yolo_dir, yaml_path = convert_to_yolo_format(test_data, output_dir)

    # 转换为 OCR 格式
    ocr_dir, csv_path = convert_to_ocr_format(test_data, output_dir)

    # 保存复核列表
    save_review_list(all_review, output_dir)

    # 保存数据映射（用于测试时追溯）
    mapping_path = Path(output_dir) / "test_data_mapping.json"
    mapping = {
        "shuffle_indices": shuffle_indices,
        "total_samples": len(test_data),
        "eye_level_count": len(eye_valid),
        "side_level_count": len(side_valid),
        "review_count": len(all_review)
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    print(f"YOLO 数据集: {yolo_dir}")
    print(f"OCR 数据集: {ocr_dir}")
    print(f"数据映射: {mapping_path}")


if __name__ == "__main__":
    main()
