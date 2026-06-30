"""
异物数据扩充脚本
将透明背景的异物素材随机贴入 Label Studio 标注的有效区域内，
生成扩充图片和对应的 Label Studio JSON 标注文件。

输出格式: Label Studio JSON（与原始标注格式一致）
后续通过 convert_labelstudio_new.py 转换为 YOLO 标签。

用法:
  python scripts/augment_debris.py \
    --dataset_dir "raw_data/track_intrusion" \
    --debris_dir  "raw_data/debris_materials" \
    --class_name  "track_region"
"""

import json
import os
import random
import math
import argparse
import uuid

# ──────────────────────────────────────────────
# 分类基准尺寸配置（面积像素，保持原始宽高比）
# ──────────────────────────────────────────────
# 确认的异物大小：石头 22px, 煤渣 30px, 水瓶 ~16x28px
# 同类异物浮动 ±10%
CATEGORY_BASE_SIZES = {
    "rock":     22,   # 石头：22x22 ≈ 484px²
    "coalslag": 30,   # 煤渣：30x30 ≈ 900px²
    "bottle":   28,   # 水瓶：~17x24 ≈ 400px²（保持长条形，scale 1.0 ≈ 33x35）
}
DEFAULT_BASE_SIZE = 22  # 未分类异物的默认基准
from pathlib import Path
from PIL import Image


# ──────────────────────────────────────────────
# 1. 解析 Label Studio JSON
# ──────────────────────────────────────────────

def parse_labelstudio_json(json_path, class_name):
    """
    解析 Label Studio JSON，提取指定类别的放置区域。

    返回: {图片文件名: [(x, y, width, height), ...]}
      坐标为百分比（0-100），与 Label Studio 一致。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for task in data:
        file_name = task.get("file_upload", "")
        if not file_name:
            image_path = task.get("data", {}).get("image", "")
            file_name = os.path.basename(image_path)

        clean_name = file_name
        if "-" in file_name:
            parts = file_name.split("-", 1)
            if len(parts[0]) == 8 and parts[0].isalnum():
                clean_name = parts[1]

        regions = []
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        for ann in annotations:
            for item in ann.get("result", []):
                if item.get("type") != "rectanglelabels":
                    continue
                labels = item.get("value", {}).get("rectanglelabels", [])
                # 支持带 _val/_test 后缀的类别名（如 carriage_rim_region_val）
                matched = any(l == class_name or l.startswith(class_name + "_") for l in labels)
                if not matched:
                    continue
                val = item["value"]
                regions.append((
                    val["x"], val["y"],
                    val["width"], val["height"]
                ))

        if regions:
            result[clean_name] = regions
            if file_name != clean_name:
                result[file_name] = regions

    return result


def find_regions_for_image(image_name, regions_map):
    """查找图片对应的放置区域，支持模糊匹配。"""
    if image_name in regions_map:
        return regions_map[image_name]

    name_no_ext = os.path.splitext(image_name)[0]
    for key in regions_map:
        if os.path.splitext(key)[0] == name_no_ext:
            return regions_map[key]

    if "-" in image_name:
        clean = image_name.split("-", 1)[1]
        if clean in regions_map:
            return regions_map[clean]

    return None


# ──────────────────────────────────────────────
# 2. 加载异物素材
# ──────────────────────────────────────────────

def load_debris_images(debris_dir, debris_filter=None, category_sizes=None):
    """
    加载异物 PNG 图片（保留 alpha 通道）。

    debris_filter: 可用的子目录名列表，如 ["rock", "bottle"]。
                   None 表示加载所有子目录。
    category_sizes: {子目录名: 目标面积} 字典。None 表示不缩放。
                    按面积缩放，保持原始宽高比。
    """
    debris_list = []
    debris_path = Path(debris_dir)

    if not debris_path.exists():
        print(f"[ERROR] 异物素材目录不存在: {debris_dir}")
        return debris_list

    if category_sizes is None:
        category_sizes = CATEGORY_BASE_SIZES

    for sub_dir in sorted(debris_path.iterdir()):
        if not sub_dir.is_dir():
            continue
        # 跳过预览目录
        if sub_dir.name == "preview":
            continue
        # 过滤子目录
        if debris_filter and sub_dir.name not in debris_filter:
            continue

        # 获取该类别的目标面积
        cat_name = sub_dir.name
        base_size_val = category_sizes.get(cat_name, DEFAULT_BASE_SIZE)
        base_area = base_size_val * base_size_val

        for img_file in sorted(sub_dir.iterdir()):
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                try:
                    img = Image.open(img_file).convert("RGBA")
                    # ±10% 随机浮动，每个素材固定一个尺寸
                    variation = random.uniform(0.9, 1.1)
                    target_area = int(base_area * variation)
                    orig_area = img.size[0] * img.size[1]
                    if orig_area > 0:
                        scale = (target_area / orig_area) ** 0.5
                        new_w = max(int(img.size[0] * scale), 5)
                        new_h = max(int(img.size[1] * scale), 5)
                        img = img.resize((new_w, new_h), Image.LANCZOS)
                    debris_list.append((img, img_file.name, cat_name))
                except Exception as e:
                    print(f"[WARN] 无法加载 {img_file}: {e}")

    # 根目录下的图片（不属于任何子目录）
    if not debris_filter:
        for img_file in sorted(debris_path.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                try:
                    img = Image.open(img_file).convert("RGBA")
                    debris_list.append((img, img_file.name, "root"))
                except Exception as e:
                    print(f"[WARN] 无法加载 {img_file}: {e}")

    categories = set(d[2] for d in debris_list)
    print(f"  加载异物素材: {len(debris_list)} 张 (类别: {', '.join(sorted(categories)) or '无'})")
    # 显示每类素材的实际尺寸
    for cat in sorted(categories):
        cat_imgs = [d for d in debris_list if d[2] == cat]
        sizes = [f"{d[0].size[0]}x{d[0].size[1]}" for d in cat_imgs[:3]]
        print(f"    {cat}: {', '.join(sizes)}{'...' if len(cat_imgs) > 3 else ''}")
    return debris_list


# ──────────────────────────────────────────────
# 3. 坐标转换与 IoU
# ──────────────────────────────────────────────

def percent_to_pixel(box_pct, img_width, img_height):
    """Label Studio 百分比 (x, y, w, h) → 像素 (x1, y1, x2, y2)。"""
    x1 = box_pct[0] / 100.0 * img_width
    y1 = box_pct[1] / 100.0 * img_height
    x2 = (box_pct[0] + box_pct[2]) / 100.0 * img_width
    y2 = (box_pct[1] + box_pct[3]) / 100.0 * img_height
    return (x1, y1, x2, y2)


def pixel_to_percent(box_px, img_width, img_height):
    """像素 (x1, y1, x2, y2) → Label Studio 百分比 (x, y, w, h)。"""
    x1, y1, x2, y2 = box_px
    x = x1 / img_width * 100.0
    y = y1 / img_height * 100.0
    w = (x2 - x1) / img_width * 100.0
    h = (y2 - y1) / img_height * 100.0
    return (x, y, w, h)


def compute_iou(box1, box2):
    """两个框 IoU。box: (x1, y1, x2, y2) 像素坐标。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ──────────────────────────────────────────────
# 4. 异物变换与贴图
# ──────────────────────────────────────────────

def random_transform_debris(debris_img, region_px, img_size, scale_range=(0.5, 1.5)):
    """
    随机缩放 + 旋转 + 在放置区域内定位。

    异物最终像素大小 = load_debris_images 中按分类基准面积缩放后的尺寸 * scale。
    同类异物 ±10% 浮动，同一素材尺寸固定。
    代码只保证异物不超出区域边界。

    返回: (transformed_img, paste_box_px) 或 None
    """
    rw = region_px[2] - region_px[0]
    rh = region_px[3] - region_px[1]

    if rw < 10 or rh < 10:
        return None

    dw, dh = debris_img.size
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w = max(int(dw * scale), 5)
    new_h = max(int(dh * scale), 5)

    angle = random.uniform(0, 360)
    rotated = debris_img.resize((new_w, new_h), Image.LANCZOS)
    rotated = rotated.rotate(angle, expand=True, resample=Image.BICUBIC)

    rw_final, rh_final = rotated.size
    if rw_final >= rw or rh_final >= rh:
        factor = min(rw / rw_final, rh / rh_final) * 0.8
        rw_final = max(int(rw_final * factor), 5)
        rh_final = max(int(rh_final * factor), 5)
        rotated = rotated.resize((rw_final, rh_final), Image.LANCZOS)

    cx = random.uniform(region_px[0] + rw_final / 2, region_px[2] - rw_final / 2)
    cy = random.uniform(region_px[1] + rh_final / 2, region_px[3] - rh_final / 2)

    paste_x = int(cx - rw_final / 2)
    paste_y = int(cy - rh_final / 2)
    paste_x = max(0, min(paste_x, img_size[0] - rw_final))
    paste_y = max(0, min(paste_y, img_size[1] - rh_final))

    paste_box = (paste_x, paste_y, paste_x + rw_final, paste_y + rh_final)
    return (rotated, paste_box)


def paste_with_alpha(base_img, overlay_img, position):
    """使用 alpha 通道合成。"""
    x, y = position
    if x < 0 or y < 0:
        return
    if overlay_img.mode == "RGBA":
        r, g, b, a = overlay_img.split()
        base_img.paste(overlay_img, (x, y), a)
    else:
        base_img.paste(overlay_img, (x, y))


# ──────────────────────────────────────────────
# 5. 生成 Label Studio JSON
# ──────────────────────────────────────────────

def make_ls_task(task_id, image_filename, image_rel_path, annotations_pct, img_width, img_height):
    """
    构建一个 Label Studio task 条目。

    annotations_pct: [(x, y, w, h, label_name), ...]  百分比坐标
    """
    result = []
    for (x, y, w, h, label) in annotations_pct:
        uid = uuid.uuid4().hex[:10]
        result.append({
            "id": uid,
            "type": "rectanglelabels",
            "value": {
                "x": round(x, 6),
                "y": round(y, 6),
                "width": round(w, 6),
                "height": round(h, 6),
                "rotation": 0,
                "rectanglelabels": [label]
            },
            "from_name": "label",
            "to_name": "image",
            "origin": "manual"
        })

    return {
        "id": task_id,
        "annotations": [{"result": result}],
        "data": {"image": image_rel_path},
        "file_upload": image_filename
    }


def save_labelstudio_json(tasks, output_path):
    """保存 Label Studio JSON。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 6. 图片名匹配
# ──────────────────────────────────────────────

def find_matching_image(label_name, image_files):
    """根据标签名找对应图片文件。"""
    name_no_ext = os.path.splitext(label_name)[0]

    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        if (name_no_ext + ext) in image_files:
            return name_no_ext + ext

    for img_name in image_files:
        img_no_ext = os.path.splitext(img_name)[0]
        if img_no_ext.endswith(name_no_ext) or name_no_ext.endswith(img_no_ext):
            return img_name
        if "-" in img_no_ext:
            clean = img_no_ext.split("-", 1)[1]
            if clean == name_no_ext:
                return img_name

    return None


# ──────────────────────────────────────────────
# 7. 主扩充流程
# ──────────────────────────────────────────────

def augment_dataset(args):
    dataset_dir = Path(args.dataset_dir)
    debris_dir = Path(args.debris_dir)
    class_name = args.class_name

    debris_images = load_debris_images(debris_dir, args.debris_filter)
    if not debris_images:
        print("[ERROR] 没有可用的异物素材，退出。")
        return

    total_aug = 0
    total_skip = 0
    summary_lines = []
    print(f"  分类基准面积: {CATEGORY_BASE_SIZES}")

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            print(f"  [SKIP] {split}: images 目录不存在")
            continue

        if split == "test" and not args.augment_test:
            print(f"  [SKIP] {split}: test 集不扩充")
            continue

        # 扫描 JSON
        json_files = list(labels_dir.glob("*.json")) if labels_dir.exists() else []
        if not json_files:
            print(f"  [SKIP] {split}: 没有 Label Studio JSON 文件")
            continue

        # 解析放置区域
        regions_map = {}
        for jf in json_files:
            parsed = parse_labelstudio_json(str(jf), class_name)
            regions_map.update(parsed)

        print(f"  [{split}] 解析到 {len(regions_map)} 张图有放置区域")

        # 扫描图片
        image_files = set()
        for f in images_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_files.add(f.name)

        # 选择扩充候选
        aug_candidates = []
        for img_name in sorted(image_files):
            if "_aug" in img_name:
                continue
            regions = find_regions_for_image(img_name, regions_map)
            if regions:
                aug_candidates.append((img_name, regions))

        num_to_aug = int(len(aug_candidates) * args.aug_ratio)
        selected = random.sample(aug_candidates, min(num_to_aug, len(aug_candidates)))

        split_skip = len(aug_candidates) - len(selected) + (len(image_files) - len(aug_candidates) - sum(1 for f in image_files if "_aug" in f))
        print(f"  [{split}] 可扩充 {len(aug_candidates)} 张，实际扩充 {len(selected)} 张")

        # 收集扩充后的 task 条目
        aug_tasks = []
        aug_idx_global = 0

        for img_name, regions in selected:
            img_path = images_dir / img_name
            try:
                base_img = Image.open(img_path).convert("RGBA")
            except Exception as e:
                print(f"    [WARN] 无法打开 {img_name}: {e}")
                continue

            img_w, img_h = base_img.size
            regions_px = [percent_to_pixel(r, img_w, img_h) for r in regions]

            # 每张原图生成 num_per_image 张扩充图，每张图异物数量服从泊松分布
            for aug_local in range(args.num_per_image):
                aug_img = base_img.copy()
                debris_boxes_px = []

                # 泊松分布 (lambda=debris_lambda)，截断到 [1, 5]
                lam = args.debris_lambda
                if lam > 0:
                    L = math.exp(-lam)
                    k, p = 0, 1.0
                    while True:
                        k += 1
                        p *= random.random()
                        if p < L:
                            break
                    n_debris = max(1, min(k - 1, 5))
                else:
                    n_debris = args.num_per_item

                placed = 0
                attempts = 0
                max_attempts = 30

                while placed < n_debris and attempts < max_attempts:
                    attempts += 1
                    debris_img, debris_name, debris_cat = random.choice(debris_images)
                    region = random.choice(regions_px)

                    result = random_transform_debris(
                        debris_img, region, (img_w, img_h),
                        scale_range=(args.scale_min, args.scale_max)
                    )
                    if result is None:
                        continue

                    transformed, paste_box = result

                    overlap = False
                    for existing_box in debris_boxes_px:
                        if compute_iou(paste_box, existing_box) > args.max_iou:
                            overlap = True
                            break
                    if overlap:
                        continue

                    paste_with_alpha(aug_img, transformed, (paste_box[0], paste_box[1]))
                    debris_boxes_px.append(paste_box)
                    placed += 1

                if placed == 0:
                    continue

                # 保存扩充图片
                name_stem = os.path.splitext(img_name)[0]
                name_ext = os.path.splitext(img_name)[1]
                aug_img_name = f"{name_stem}_aug{aug_local}{name_ext}"
                aug_img.convert("RGB").save(images_dir / aug_img_name)

                # 构建 Label Studio annotations
                annotations_pct = []

                # 原始 region 标注
                for r in regions:
                    annotations_pct.append((r[0], r[1], r[2], r[3], class_name))

                # 异物标注
                for box_px in debris_boxes_px:
                    x, y, w, h = pixel_to_percent(box_px, img_w, img_h)
                    annotations_pct.append((x, y, w, h, "debris"))

                task = make_ls_task(
                    task_id=10000 + aug_idx_global,
                    image_filename=aug_img_name,
                    image_rel_path=f"/data/upload/{aug_img_name}",
                    annotations_pct=annotations_pct,
                    img_width=img_w,
                    img_height=img_h
                )
                aug_tasks.append(task)
                aug_idx_global += 1

        # 保存扩充 JSON
        if aug_tasks:
            json_name = f"augmented_debris_{split}.json"
            json_path = labels_dir / json_name
            save_labelstudio_json(aug_tasks, str(json_path))
            print(f"  [{split}] 保存扩充 JSON: {json_name} ({len(aug_tasks)} 条)")

        total_aug += len(aug_tasks)
        total_skip += split_skip
        summary_lines.append(f"{split}: 扩充 {len(aug_tasks)} 张")

    # 汇总
    summary = [
        "=" * 50,
        "异物数据扩充汇总",
        "=" * 50,
        f"数据集: {dataset_dir}",
        f"异物素材: {debris_dir}",
        f"放置区域类别: {class_name}",
        f"每张图生成: {args.num_per_image} 张扩充图",
        f"异物数量分布: Poisson(λ={args.debris_lambda})" if args.debris_lambda > 0 else f"异物数量: 固定 {args.num_per_item}",
        f"扩充比例: {args.aug_ratio}",
        "-" * 50,
    ] + summary_lines + [
        "-" * 50,
        f"总计扩充: {total_aug} 张",
        "=" * 50,
    ]

    print()
    for line in summary:
        print(line)

    summary_path = dataset_dir / "augment_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\n汇总已保存: {summary_path}")


# ──────────────────────────────────────────────
# 8. 预览模式：不同尺寸对比
# ──────────────────────────────────────────────

def preview_mode(args):
    """
    预览模式：对少量图片生成不同缩放比例的异物贴图对比图。
    每张原图生成一组对比图，方便确定合适的缩放参数。
    """
    dataset_dir = Path(args.dataset_dir)
    debris_dir = Path(args.debris_dir)
    class_name = args.class_name

    debris_images = load_debris_images(debris_dir, args.debris_filter)
    if not debris_images:
        print("[ERROR] 没有可用的异物素材，退出。")
        return

    print(f"  分类基准面积: {CATEGORY_BASE_SIZES}")

    # 预览用的缩放比例档位
    if args.scale_min != 0.5 or args.scale_max != 1.5:
        # 用户指定了自定义 scale，只用该范围预览
        scale_presets = [
            (args.scale_min, args.scale_max, f"自定义-{args.scale_min}-{args.scale_max}"),
        ]
    else:
        scale_presets = [
            (0.1, 0.3, "S-小"),
            (0.3, 0.6, "M-中"),
            (0.6, 1.0, "L-大"),
            (1.0, 1.5, "XL-特大"),
            (1.5, 2.5, "XXL-超大"),
            (2.5, 4.0, "XXX-极大"),
        ]

    # 找一张有标注区域的图
    sample_img = None
    sample_regions = None

    for split in ["train", "val"]:
        split_dir = dataset_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        json_files = list(labels_dir.glob("*.json"))
        if not json_files:
            continue

        regions_map = {}
        for jf in json_files:
            parsed = parse_labelstudio_json(str(jf), class_name)
            regions_map.update(parsed)

        for f in sorted(images_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png") and "_aug" not in f.name:
                regions = find_regions_for_image(f.name, regions_map)
                if regions:
                    sample_img = f
                    sample_regions = regions
                    break
        if sample_img:
            break

    if not sample_img:
        print("[ERROR] 找不到有标注区域的图片，无法生成预览。")
        return

    print(f"  预览图片: {sample_img.name}")
    print(f"  放置区域: {sample_regions}")

    base_img = Image.open(sample_img).convert("RGBA")
    img_w, img_h = base_img.size
    regions_px = [percent_to_pixel(r, img_w, img_h) for r in sample_regions]

    # 创建输出目录
    preview_dir = dataset_dir / "preview"
    preview_dir.mkdir(exist_ok=True)

    # 生成原图
    base_img.convert("RGB").save(preview_dir / "00_original.jpg")

    # 每个档位生成多张对比
    for scale_min, scale_max, label in scale_presets:
        for i in range(3):  # 每档 3 张
            aug_img = base_img.copy()
            debris_boxes = []

            placed = 0
            for _ in range(20):
                debris_img, debris_name, debris_cat = random.choice(debris_images)
                region = random.choice(regions_px)

                result = random_transform_debris(
                    debris_img, region, (img_w, img_h),
                    scale_range=(scale_min, scale_max)
                )
                if result is None:
                    continue

                transformed, paste_box = result

                overlap = False
                for existing_box in debris_boxes:
                    if compute_iou(paste_box, existing_box) > args.max_iou:
                        overlap = True
                        break
                if overlap:
                    continue

                paste_with_alpha(aug_img, transformed, (paste_box[0], paste_box[1]))
                debris_boxes.append(paste_box)
                placed += 1
                if placed >= 2:
                    break

            out_name = f"{label}_{i+1}.jpg"
            aug_img.convert("RGB").save(preview_dir / out_name)
            print(f"  生成: {out_name} (scale {scale_min}-{scale_max}, {placed} 个异物)")

    print(f"\n预览图已保存到: {preview_dir}")
    print("请查看图片确定合适的缩放比例，然后用 --scale_min 和 --scale_max 参数运行正式扩充。")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="异物数据扩充脚本")
    parser.add_argument("--dataset_dir", required=True, help="数据集根目录")
    parser.add_argument("--debris_dir", required=True, help="异物素材目录")
    parser.add_argument("--class_name", default="track_region",
                        help="Label Studio 中放置区域的类别名 (default: track_region)")
    parser.add_argument("--debris_filter", nargs="*", default=None,
                        help="只使用指定的素材子目录名，如 rock bottle (default: 全部)")
    parser.add_argument("--num_per_image", type=int, default=2,
                        help="每张原图生成几张扩充图 (default: 2)")
    parser.add_argument("--debris_lambda", type=float, default=2.5,
                        help="泊松分布 λ，控制每张图异物数量 (default: 2.5, 均值≈2-3个)")
    parser.add_argument("--num_per_item", type=int, default=2,
                        help="debris_lambda=0 时的固定异物数 (default: 2)")
    parser.add_argument("--aug_ratio", type=float, default=0.5,
                        help="扩充比例 0-1 (default: 0.5)")
    parser.add_argument("--max_iou", type=float, default=0.3,
                        help="异物间最大 IoU (default: 0.3)")
    parser.add_argument("--scale_min", type=float, default=0.5,
                        help="异物最小缩放比 (default: 0.5)")
    parser.add_argument("--scale_max", type=float, default=1.5,
                        help="异物最大缩放比 (default: 1.5)")
    parser.add_argument("--augment_test", action="store_true",
                        help="是否对 test 集也做扩充 (default: False)")
    parser.add_argument("--preview", action="store_true",
                        help="预览模式：生成不同尺寸对比图，不做正式扩充")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 50)
    print("异物数据扩充脚本")
    print("=" * 50)
    print(f"\n数据集: {args.dataset_dir}")
    print(f"异物素材: {args.debris_dir}")
    if args.debris_filter:
        print(f"素材过滤: {args.debris_filter}")
    print(f"放置区域类别: {args.class_name}")

    if args.preview:
        print(f"\n--- 预览模式 ---\n")
        preview_mode(args)
    else:
        print(f"每张图生成: {args.num_per_image} 张扩充图")
        if args.debris_lambda > 0:
            print(f"异物数量分布: Poisson(λ={args.debris_lambda})，截断 [1,5]")
        else:
            print(f"异物数量: 固定 {args.num_per_item}")
        print(f"扩充比例: {args.aug_ratio}")
        print(f"缩放范围: {args.scale_min} ~ {args.scale_max}\n")
        augment_dataset(args)


if __name__ == "__main__":
    main()
