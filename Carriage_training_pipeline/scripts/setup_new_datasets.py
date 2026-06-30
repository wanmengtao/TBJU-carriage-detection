"""
从 raw_data/ 复制图片到 datasets/（含扩充图），并生成 dataset.yaml。

应在 augment_debris.py 运行后执行，确保扩充图也被复制。

用法: python scripts/setup_new_datasets.py
"""

import os
import shutil
from pathlib import Path

# 项目根目录（自动推导）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 源目录 → 目标目录映射
DATASETS = [
    {
        "src": str(PROJECT_ROOT / "raw_data" / "carriage_rim_debris"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "carriage_rim_debris_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "carriage_rim_debris_detection_test"),
        "task": "车厢沿异物检测",
        "nc": 2,
        "names": ["region", "debris"],
    },
    {
        "src": str(PROJECT_ROOT / "raw_data" / "door_state"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "door_state_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "door_state_detection_test"),
        "task": "车门状态识别",
        "nc": 1,
        "names": ["region"],
    },
    {
        "src": str(PROJECT_ROOT / "raw_data" / "track_intrusion"),
        "dst_trainval": str(PROJECT_ROOT / "datasets" / "output" / "track_intrusion_detection"),
        "dst_test": str(PROJECT_ROOT / "datasets" / "test_output" / "track_intrusion_detection_test"),
        "task": "轨道异物侵限检测",
        "nc": 2,
        "names": ["region", "debris"],
    },
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def clear_images(directory):
    """清空目录中的图片文件。"""
    if not directory.exists():
        return 0
    count = 0
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            f.unlink()
            count += 1
    return count


def copy_images(src_dir, dst_dir):
    """复制图片到目标目录，返回复制数量。"""
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(f, dst_dir / f.name)
            count += 1
    return count


def setup_dataset(ds):
    """为一个数据集同步图片（含扩充图）。"""
    src = Path(ds["src"])
    dst_tv = Path(ds["dst_trainval"])
    dst_test = Path(ds["dst_test"])

    if not src.exists():
        print(f"  [SKIP] 源目录不存在: {src}")
        return 0

    total_images = 0

    # train / val → dst_trainval/{split}/images/
    for split in ["train", "val"]:
        src_images = src / split / "images"
        dst_images = dst_tv / split / "images"
        dst_labels = dst_tv / split / "labels"

        if not src_images.exists():
            continue

        # 清空旧图片，重新复制
        cleared = clear_images(dst_images)
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        count = copy_images(src_images, dst_images)
        total_images += count
        status = f"{count} 张已复制"
        if cleared > 0:
            status += f"（清除了 {cleared} 张旧图）"
        print(f"    {split}: {status}")

    # test → dst_test/images/
    src_test_images = src / "test" / "images"
    dst_test_images = dst_test / "images"
    dst_test_labels = dst_test / "labels"

    if src_test_images.exists():
        cleared = clear_images(dst_test_images)
        dst_test_images.mkdir(parents=True, exist_ok=True)
        dst_test_labels.mkdir(parents=True, exist_ok=True)

        count = copy_images(src_test_images, dst_test_images)
        total_images += count
        status = f"{count} 张已复制"
        if cleared > 0:
            status += f"（清除了 {cleared} 张旧图）"
        print(f"    test: {status}")

    # 生成 dataset.yaml（train/val 目录下）
    names_str = str(ds["names"])
    yaml_content = f"""path: {dst_tv.resolve()}
train: train/images
val: val/images
test: test/images

nc: {ds['nc']}
names: {names_str}
"""
    yaml_path = dst_tv / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 生成 test dataset.yaml
    if dst_test.exists():
        test_yaml_content = f"""path: {dst_test.resolve()}
train: images
val: images
test: images

nc: {ds['nc']}
names: {names_str}
"""
        test_yaml_path = dst_test / "dataset.yaml"
        with open(test_yaml_path, "w", encoding="utf-8") as f:
            f.write(test_yaml_content)

    return total_images


def main():
    print("=" * 50)
    print("同步图片到数据集标注目录（含扩充图）")
    print("=" * 50)

    for ds in DATASETS:
        print(f"\n[{ds['task']}] {Path(ds['dst_trainval']).name}")
        count = setup_dataset(ds)
        print(f"  共 {count} 张图片")

    # 汇总
    print(f"\n{'='*50}")
    print("同步完成。目录结构:")
    for ds in DATASETS:
        dst_tv = Path(ds["dst_trainval"])
        dst_test = Path(ds["dst_test"])
        print(f"\n  {dst_tv.name}/")
        for split in ["train", "val"]:
            img_dir = dst_tv / split / "images"
            lbl_dir = dst_tv / split / "labels"
            img_count = len(list(img_dir.iterdir())) if img_dir.exists() else 0
            lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            print(f"    {split}/images: {img_count} 张, labels: {lbl_count} 个 txt")
        img_dir = dst_test / "images"
        lbl_dir = dst_test / "labels"
        img_count = len(list(img_dir.iterdir())) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        print(f"    test/images: {img_count} 张, labels: {lbl_count} 个 txt")
    print(f"\n{'='*50}")


if __name__ == "__main__":
    main()
