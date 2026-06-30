"""
Label Studio JSON → YOLO 标签转换脚本（异物/车门数据集专用）

将 raw_data 中三个数据集的 Label Studio JSON 转换为 YOLO txt 标签，
输出到 datasets/ 对应目录。

每个 split 的 labels/ 目录下可能有两个 JSON：
  1. 手动标注的 JSON（region 区域框）
  2. augment_debris.py 生成的 augmented_debris_*.json（region + debris 框）
脚本会合并处理所有 JSON，生成统一的 YOLO 标签。

用法:
  python scripts/convert_labelstudio_new.py
"""

import json
import os
from pathlib import Path

# 项目根目录（自动推导）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 数据集配置：源目录、输出目录、region 类别名
DATASETS = [
    {
        "name": "carriage_rim_debris",
        "src": str(PROJECT_ROOT / "raw_data" / "carriage_rim_debris"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "carriage_rim_debris_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "carriage_rim_debris_detection_test"),
        "region_class": "carriage_rim_region",
    },
    {
        "name": "track_intrusion",
        "src": str(PROJECT_ROOT / "raw_data" / "track_intrusion"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "track_intrusion_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "track_intrusion_detection_test"),
        "region_class": "track_region",
    },
    {
        "name": "door_state",
        "src": str(PROJECT_ROOT / "raw_data" / "door_state"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "door_state_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "door_state_detection_test"),
        "region_class": "door_region",
    },
]

# 类别映射：Label Studio 标签名 → YOLO class_id
CLASS_MAP = {
    "track_region": 0,
    "track_region_val": 0,
    "track_region_test": 0,
    "carriage_rim_region": 0,
    "carriage_rim_region_val": 0,
    "carriage_rim_region_test": 0,
    "door_region": 0,
    "door_region_val": 0,
    "door_region_test": 0,
    "region": 0,
    "debris": 1,
}

CLASS_NAMES = ["region", "debris"]


def parse_ls_json(json_path):
    """
    解析 Label Studio JSON，返回 {图片文件名: [(x, y, w, h, class_name), ...]}
    坐标为百分比（0-100）。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for task in data:
        file_name = task.get("file_upload", "")
        if not file_name:
            image_path = task.get("data", {}).get("image", "")
            file_name = os.path.basename(image_path)

        if not file_name:
            continue

        # 剥离 Label Studio UUID 前缀（如 "015dc529-0006_S_f0349.jpg" → "0006_S_f0349.jpg"）
        if "-" in file_name:
            parts = file_name.split("-", 1)
            if len(parts[0]) == 8 and parts[0].isalnum():
                file_name = parts[1]

        annotations = task.get("annotations", [])
        if not annotations:
            continue

        boxes = []
        for ann in annotations:
            for item in ann.get("result", []):
                if item.get("type") != "rectanglelabels":
                    continue
                val = item.get("value", {})
                labels = val.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0]
                boxes.append((
                    val["x"], val["y"],
                    val["width"], val["height"],
                    label
                ))

        if boxes:
            result[file_name] = boxes

    return result


def ls_to_yolo(x, y, w, h):
    """Label Studio 百分比 (x, y, w, h) → YOLO 归一化 (x_center, y_center, w, h)。"""
    x_center = (x + w / 2.0) / 100.0
    y_center = (y + h / 2.0) / 100.0
    w_norm = w / 100.0
    h_norm = h / 100.0
    return x_center, y_center, w_norm, h_norm


def match_image_to_label(label_name, image_files):
    """
    根据 JSON 中的文件名找到对应的图片文件。
    支持 UUID 前缀模糊匹配。
    """
    # 精确匹配
    if label_name in image_files:
        return label_name

    # 去扩展名匹配
    name_stem = os.path.splitext(label_name)[0]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        if name_stem + ext in image_files:
            return name_stem + ext

    # UUID 前缀匹配：JSON 中是 DJI_xxx.jpg，图片是 abcdef12-DJI_xxx.jpg
    for img_name in image_files:
        img_stem = os.path.splitext(img_name)[0]
        # 图片有 UUID 前缀，去掉后匹配
        if "-" in img_stem:
            clean = img_stem.split("-", 1)[1]
            if clean == name_stem or clean + os.path.splitext(img_name)[1] == label_name:
                return img_name
        # JSON 名有 UUID 前缀，去掉后匹配
        if "-" in name_stem:
            clean_label = name_stem.split("-", 1)[1]
            if img_stem == clean_label:
                return img_name

    return None


def convert_dataset(ds_config):
    """转换单个数据集的所有 split。"""
    name = ds_config["name"]
    src_dir = Path(ds_config["src"])
    region_class = ds_config["region_class"]

    print(f"\n{'='*50}")
    print(f"[{name}] region 类别: {region_class}")
    print(f"{'='*50}")

    total_converted = 0
    total_skipped = 0

    for split in ["train", "val", "test"]:
        src_labels = src_dir / split / "labels"
        src_images = src_dir / split / "images"

        if not src_labels.exists() or not src_images.exists():
            print(f"  [{split}] 跳过（目录不存在）")
            continue

        # 确定输出目录
        if split == "test":
            dst_dir = Path(ds_config["dst_test"])
        else:
            dst_dir = Path(ds_config["dst_trainval"])

        dst_labels = dst_dir / "labels" if split == "test" else dst_dir / split / "labels"

        dst_labels.mkdir(parents=True, exist_ok=True)

        # 扫描所有 JSON 文件
        json_files = list(src_labels.glob("*.json"))
        if not json_files:
            print(f"  [{split}] 跳过（无 JSON 文件）")
            continue

        # 合并解析所有 JSON（手动标注 + augment_debris 生成的）
        all_annotations = {}
        json_info = []
        for jf in sorted(json_files):
            parsed = parse_ls_json(str(jf))
            all_annotations.update(parsed)
            json_info.append(f"{jf.name}({len(parsed)} 条)")

        print(f"  [{split}] JSON: {', '.join(json_info)}")

        # 从源目录扫描图片（包含原始图 + augment_debris 生成的扩充图）
        image_files = set()
        for f in src_images.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_files.add(f.name)

        # 为每张图片生成 YOLO 标签
        converted = 0
        skipped = 0
        skipped_images = []

        for img_name in sorted(image_files):
            # 查找该图片的标注
            matched_key = match_image_to_label(img_name, all_annotations)
            if matched_key is None:
                skipped += 1
                skipped_images.append(img_name)
                continue

            boxes = all_annotations[matched_key]

            # 生成 YOLO 行
            yolo_lines = []
            for (x, y, w, h, label) in boxes:
                class_id = CLASS_MAP.get(label)
                if class_id is None:
                    print(f"    [WARN] 未知类别 '{label}'，跳过 ({img_name})")
                    continue
                xc, yc, wn, hn = ls_to_yolo(x, y, w, h)
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if not yolo_lines:
                skipped += 1
                skipped_images.append(img_name)
                continue

            # 写入 txt
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = dst_labels / txt_name
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            converted += 1

        if skipped_images:
            print(f"  [{split}] 跳过的图片（无匹配标注）:")
            for s in skipped_images[:10]:
                print(f"    - {s}")
            if len(skipped_images) > 10:
                print(f"    ... 共 {len(skipped_images)} 张")

        total_converted += converted
        total_skipped += skipped
        print(f"  [{split}] 转换: {converted} 张, 跳过: {skipped} 张（无标注）")

    print(f"\n  [{name}] 总计: 转换 {total_converted} 张, 跳过 {total_skipped} 张")
    return total_converted, total_skipped


def main():
    print("=" * 50)
    print("Label Studio JSON → YOLO 标签转换")
    print("（异物/车门数据集专用）")
    print("=" * 50)

    grand_total = 0
    grand_skip = 0

    for ds in DATASETS:
        converted, skipped = convert_dataset(ds)
        grand_total += converted
        grand_skip += skipped

    print(f"\n{'='*50}")
    print(f"全部完成: 共转换 {grand_total} 张, 跳过 {grand_skip} 张")
    print(f"{'='*50}")

    # 生成 summary
    print("\n输出目录:")
    for ds in DATASETS:
        tv = Path(ds["dst_trainval"])
        te = Path(ds["dst_test"])
        for split in ["train", "val"]:
            d = tv / split / "labels"
            count = len(list(d.glob("*.txt"))) if d.exists() else 0
            print(f"  {ds['name']}/{split}: {count} 个 txt")
        d = te / "labels"
        count = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {ds['name']}/test: {count} 个 txt")


if __name__ == "__main__":
    main()
