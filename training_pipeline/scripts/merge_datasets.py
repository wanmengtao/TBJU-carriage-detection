"""
数据集合并脚本
将车号检测、车厢沿异物、轨道异物侵限、车门状态四个数据集合并为统一数据集。

统一类别映射：
  0: TBJU_region        (车号检测)
  1: carriage_rim_region (车厢沿区域)
  2: carriage_rim_debris (车厢沿异物)
  3: track_region        (轨道区域)
  4: track_intrusion_debris (轨道异物)
  5: door_region         (车门区域)

用法:
  python scripts/merge_datasets.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "output"
MERGED_DIR = PROJECT_ROOT / "datasets" / "merged_detection"

# 源数据集配置：(目录名, split, 原始class_id → 新class_id)
DATASETS = [
    # 车号检测 - 平视
    {
        "dir": "wagon_number_detection_平视",
        "class_map": {0: 0},  # TBJU_region → 0
    },
    # 车号检测 - 侧视
    {
        "dir": "wagon_number_detection_侧视",
        "class_map": {0: 0},  # TBJU_region → 0
    },
    # 车厢沿异物
    {
        "dir": "carriage_rim_debris_detection",
        "class_map": {0: 1, 1: 2},  # region → 1, debris → 2
    },
    # 轨道异物侵限
    {
        "dir": "track_intrusion_detection",
        "class_map": {0: 3, 1: 4},  # region → 3, debris → 4
    },
    # 车门状态
    {
        "dir": "door_state_detection",
        "class_map": {0: 5},  # region → 5
    },
]

TEST_DIR = PROJECT_ROOT / "datasets" / "test_output"
TEST_DATASETS = [
    {
        "dir": "wagon_number_detection_test",
        "class_map": {0: 0},
        "subdir": "test",
    },
    {
        "dir": "carriage_rim_debris_detection_test",
        "class_map": {0: 1, 1: 2},
    },
    {
        "dir": "track_intrusion_detection_test",
        "class_map": {0: 3, 1: 4},
    },
    {
        "dir": "door_state_detection_test",
        "class_map": {0: 5},
    },
]

UNIFIED_NAMES = [
    "TBJU_region",
    "carriage_rim_region",
    "carriage_rim_debris",
    "track_region",
    "track_intrusion_debris",
    "door_region",
]


def remap_label(src_label_path, class_map):
    """读取原始 YOLO 标签，重新映射 class_id。"""
    new_lines = []
    with open(src_label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = class_map.get(old_id)
            if new_id is not None:
                new_lines.append(f"{new_id} {' '.join(parts[1:])}")
    return new_lines


def copy_and_remap(src_dir, dst_dir, class_map, split, subdir=None):
    """复制图片并重新映射标签。"""
    if subdir:
        src_images = src_dir / subdir / "images"
        src_labels = src_dir / subdir / "labels"
    else:
        # 先尝试 {split}/images 结构，再尝试直接 images/ 结构
        src_images = src_dir / split / "images"
        src_labels = src_dir / split / "labels"
        if not src_images.exists():
            src_images = src_dir / "images"
            src_labels = src_dir / "labels"

    dst_images = dst_dir / split / "images"
    dst_labels = dst_dir / split / "labels"

    if not src_images.exists():
        return 0

    count = 0
    for img_file in sorted(src_images.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # 复制图片
        dst_img = dst_images / img_file.name
        if not dst_img.exists():
            shutil.copy2(img_file, dst_img)

        # 处理标签
        label_file = src_labels / (img_file.stem + ".txt")
        if label_file.exists():
            new_lines = remap_label(label_file, class_map)
            if new_lines:
                dst_label = dst_labels / (img_file.stem + ".txt")
                with open(dst_label, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                count += 1

    return count


def main():
    print("=" * 60)
    print("数据集合并脚本")
    print("=" * 60)

    # 清空目标目录
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            target = MERGED_DIR / split / sub
            if target.exists():
                for f in target.iterdir():
                    f.unlink()

    total = 0

    # 合并 train/val
    for ds in DATASETS:
        src_dir = OUTPUT_DIR / ds["dir"]
        for split in ["train", "val"]:
            count = copy_and_remap(src_dir, MERGED_DIR, ds["class_map"], split)
            if count > 0:
                print(f"  {ds['dir']}/{split}: {count} 张")
                total += count

    # 合并 test
    for ds in TEST_DATASETS:
        src_dir = TEST_DIR / ds["dir"]
        subdir = ds.get("subdir")
        count = copy_and_remap(src_dir, MERGED_DIR, ds["class_map"], "test", subdir)
        if count > 0:
            print(f"  {ds['dir']}/test: {count} 张")
            total += count

    # 生成 dataset.yaml（使用绝对路径，自动适配当前机器）
    yaml_content = f"""path: {MERGED_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(UNIFIED_NAMES)}
names: {UNIFIED_NAMES}
"""
    yaml_path = MERGED_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 统计
    print(f"\n{'=' * 60}")
    print(f"合并完成，共 {total} 张图片")
    print(f"输出目录: {MERGED_DIR}")
    print(f"类别数: {len(UNIFIED_NAMES)}")
    print(f"类别: {UNIFIED_NAMES}")
    print(f"{'=' * 60}")

    # 验证
    print("\n验证:")
    for split in ["train", "val", "test"]:
        img_count = len(list((MERGED_DIR / split / "images").iterdir()))
        lbl_count = len(list((MERGED_DIR / split / "labels").glob("*.txt")))
        print(f"  {split}: {img_count} 图片, {lbl_count} 标签")


if __name__ == "__main__":
    main()
