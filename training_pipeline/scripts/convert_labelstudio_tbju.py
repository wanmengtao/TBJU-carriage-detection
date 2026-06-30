#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label Studio JSON → YOLO检测数据集 + OCR识别数据集 转换脚本

输入：Label Studio导出的JSON文件
输出：
  1. YOLO检测数据集（图片 + txt标签）
  2. OCR识别数据集（crops + labels.csv + review_list.csv + preview）

用法：
  python scripts/convert_labelstudio_tbju.py --dataset_dirs raw_data/eye_level raw_data/side_view --output datasets/output
"""

import json
import csv
import os
import re
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("请安装Pillow库: pip install Pillow")
    exit(1)


# ==================== 配置 ====================

VALID_PREFIXES = ('TBJU', 'TBCU')
CAR_NUMBER_PATTERN = re.compile(r'^(TBJU|TBCU)\d{7}$')
IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')


# ==================== 工具函数 ====================

def clean_filename(filename):
    """去掉文件名中的UUID前缀（8位字符+横杠）"""
    name = os.path.splitext(filename)[0]
    if '-' in name and len(name.split('-')[0]) == 8:
        return '-'.join(name.split('-')[1:])
    return name


def normalize_text(text):
    """标准化车号文本"""
    if not text:
        return ''
    return text.replace(' ', '').replace('-', '').replace('[', '').replace(']', '').upper()


def is_valid_car_number(text):
    """检查文本是否符合车号格式"""
    return bool(CAR_NUMBER_PATTERN.match(text))


def find_local_image(file_upload, images_dir):
    """在本地图片目录中查找图片"""
    # 1. 尝试用file_upload原名匹配
    for ext in IMAGE_FORMATS:
        candidate = os.path.join(images_dir, file_upload)
        if os.path.exists(candidate):
            return candidate, file_upload

    # 2. 去掉UUID前缀
    clean_name = clean_filename(file_upload)
    for ext in IMAGE_FORMATS:
        candidate = os.path.join(images_dir, clean_name + ext)
        if os.path.exists(candidate):
            return candidate, clean_name + ext

    # 3. 按文件名后缀模糊匹配
    target_stem = os.path.splitext(clean_name)[0]
    for f in os.listdir(images_dir):
        if f.lower().endswith(IMAGE_FORMATS):
            if os.path.splitext(f)[0] == target_stem:
                return os.path.join(images_dir, f), f

    return None, None


def parse_json_annotations(json_path):
    """解析Label Studio JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = defaultdict(list)

    for item in data:
        file_upload = item.get('file_upload', '')
        image_name = clean_filename(file_upload)

        for ann in item.get('annotations', []):
            boxes = {}
            texts = {}

            for res in ann.get('result', []):
                res_id = res.get('id')
                res_type = res.get('type')
                value = res.get('value', {})

                if res_type == 'rectanglelabels':
                    boxes[res_id] = {
                        'x': value.get('x', 0),
                        'y': value.get('y', 0),
                        'width': value.get('width', 0),
                        'height': value.get('height', 0),
                        'label': value.get('rectanglelabels', [''])[0]
                    }
                elif res_type == 'textarea':
                    text_arr = value.get('text', [])
                    if text_arr:
                        texts[res_id] = text_arr[0]

            # 配对框和文本
            for box_id, box_info in boxes.items():
                text = texts.get(box_id, '')
                text_normalized = normalize_text(text)

                annotations[image_name].append({
                    'bbox': box_info,
                    'text_raw': text,
                    'text_normalized': text_normalized,
                    'is_valid': is_valid_car_number(text_normalized),
                    'file_upload': file_upload
                })

    return dict(annotations)


# ==================== YOLO数据集生成 ====================

def generate_yolo_dataset(annotations, images_dir, output_dir, split_name):
    """生成YOLO检测数据集"""
    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    count = 0
    for image_name, ann_list in annotations.items():
        # 查找本地图片（使用原始file_upload名进行匹配）
        file_upload = ann_list[0].get('file_upload', '') if ann_list else ''
        local_path, actual_name = None, None
        if file_upload:
            local_path, actual_name = find_local_image(file_upload, images_dir)
        if not local_path:
            for ext in IMAGE_FORMATS:
                local_path, actual_name = find_local_image(image_name + ext, images_dir)
                if local_path:
                    break

        if local_path is None:
            continue

        # 复制图片
        dst_img = os.path.join(images_out, actual_name)
        if not os.path.exists(dst_img):
            shutil.copy2(local_path, dst_img)

        # 生成YOLO标签
        if ann_list:
            label_path = os.path.join(labels_out, os.path.splitext(actual_name)[0] + '.txt')
            with open(label_path, 'w', encoding='utf-8') as f:
                for ann in ann_list:
                    bbox = ann['bbox']
                    x_center = (bbox['x'] + bbox['width'] / 2) / 100
                    y_center = (bbox['y'] + bbox['height'] / 2) / 100
                    w = bbox['width'] / 100
                    h = bbox['height'] / 100
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        count += 1

    return count


# ==================== OCR数据集生成 ====================

def generate_ocr_dataset(annotations, images_dir, output_dir, split_name):
    """生成OCR识别数据集"""
    crops_dir = os.path.join(output_dir, 'crops')
    preview_dir = os.path.join(output_dir, 'preview')
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    labels_csv_path = os.path.join(output_dir, 'labels.csv')
    review_csv_path = os.path.join(output_dir, 'review_list.csv')

    labels_rows = []
    review_rows = []
    crop_count = 0

    for image_name, ann_list in annotations.items():
        # 查找本地图片（使用原始file_upload名进行匹配）
        file_upload = ann_list[0].get('file_upload', '') if ann_list else ''
        local_path, actual_name = None, None
        if file_upload:
            local_path, actual_name = find_local_image(file_upload, images_dir)
        if not local_path:
            for ext in IMAGE_FORMATS:
                local_path, actual_name = find_local_image(image_name + ext, images_dir)
                if local_path:
                    break

        if local_path is None:
            continue

        # 读取原图
        img = Image.open(local_path)
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        box_idx = 0

        for ann in ann_list:
            bbox = ann['bbox']
            text_raw = ann['text_raw']
            text_norm = ann['text_normalized']
            is_valid = ann['is_valid']

            # 计算像素坐标
            x1 = int(bbox['x'] / 100 * img_w)
            y1 = int(bbox['y'] / 100 * img_h)
            x2 = int((bbox['x'] + bbox['width']) / 100 * img_w)
            y2 = int((bbox['y'] + bbox['height']) / 100 * img_h)

            # 裁剪并保存crops
            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{os.path.splitext(actual_name)[0]}_box{box_idx}.jpg"
            crop.save(os.path.join(crops_dir, crop_name), quality=95)

            # 写入labels.csv
            if is_valid:
                labels_rows.append({
                    'crop_name': crop_name,
                    'original_image': actual_name,
                    'text': text_norm,
                    'type': 'TBJU_region',
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'split': split_name
                })
            else:
                review_rows.append({
                    'original_image': actual_name,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'text': text_raw,
                    'reason': 'invalid_text' if text_raw else 'missing_text'
                })

            # 画框和文字
            color = 'green' if is_valid else 'red'
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            display_text = text_norm if text_norm else '[missing]'
            try:
                draw.text((x1, y1 - 15), display_text, fill=color)
            except:
                pass

            box_idx += 1
            crop_count += 1

        # 保存preview图
        preview_path = os.path.join(preview_dir, f"{os.path.splitext(actual_name)[0]}_preview.jpg")
        img.save(preview_path, quality=95)

    # 写入CSV文件
    if labels_rows:
        with open(labels_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['crop_name', 'original_image', 'text', 'type', 'x1', 'y1', 'x2', 'y2', 'split'])
            writer.writeheader()
            writer.writerows(labels_rows)

    if review_rows:
        with open(review_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['original_image', 'x1', 'y1', 'x2', 'y2', 'text', 'reason'])
            writer.writeheader()
            writer.writerows(review_rows)

    return crop_count, len(labels_rows), len(review_rows)


# ==================== 主处理函数 ====================

def process_dataset(dataset_dir, output_root, seed=42):
    """处理单个数据集目录"""
    dataset_name = os.path.basename(dataset_dir)
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    stats = {
        'total_tasks': 0,
        'matched_images': 0,
        'unmatched_images': 0,
        'total_boxes': 0,
        'valid_tbju_boxes': 0,
        'review_count': 0,
        'train_images': 0,
        'val_images': 0,
        'train_crops': 0,
        'val_crops': 0,
    }

    # 收集所有标注
    train_annotations = {}
    val_annotations = {}

    # 处理train和val
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"  跳过 {split}：目录不存在")
            continue

        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"  跳过 {split}：images或labels目录不存在")
            continue

        # 找到JSON文件
        json_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
        if not json_files:
            print(f"  跳过 {split}：没有找到JSON文件")
            continue

        # 解析JSON
        for json_file in json_files:
            json_path = os.path.join(labels_dir, json_file)
            print(f"  解析: {json_file}")

            annotations = parse_json_annotations(json_path)
            stats['total_tasks'] += len(annotations)

            # 统计标注
            for img_name, ann_list in annotations.items():
                for ann in ann_list:
                    stats['total_boxes'] += 1
                    if ann['is_valid']:
                        stats['valid_tbju_boxes'] += 1
                    else:
                        stats['review_count'] += 1

            if split == 'train':
                train_annotations.update(annotations)
            else:
                val_annotations.update(annotations)

        # 检查图片匹配（使用find_local_image进行鲁棒匹配）
        matched = 0
        unmatched = 0
        for json_img, ann_list in annotations.items():
            file_upload = ann_list[0].get('file_upload', '') if ann_list else ''
            local_path, _ = find_local_image(file_upload, images_dir) if file_upload else (None, None)
            if not local_path:
                local_path, _ = find_local_image(json_img, images_dir)
            if local_path:
                matched += 1
            else:
                for ext in IMAGE_FORMATS:
                    local_path, _ = find_local_image(json_img + ext, images_dir)
                    if local_path:
                        matched += 1
                        break
                else:
                    unmatched += 1

        stats['matched_images'] += matched
        stats['unmatched_images'] += unmatched

        if unmatched:
            print(f"  [警告] {unmatched} 张图片在images目录中找不到")

    # 生成YOLO数据集
    print(f"\n  生成YOLO检测数据集...")
    yolo_output = os.path.join(output_root, f'wagon_number_detection_{dataset_name}')

    train_yolo = os.path.join(yolo_output, 'train')
    stats['train_images'] = generate_yolo_dataset(
        train_annotations, os.path.join(dataset_dir, 'train', 'images'),
        train_yolo, 'train'
    )
    print(f"    Train: {stats['train_images']} 张图片")

    val_yolo = os.path.join(yolo_output, 'val')
    stats['val_images'] = generate_yolo_dataset(
        val_annotations, os.path.join(dataset_dir, 'val', 'images'),
        val_yolo, 'val'
    )
    print(f"    Val: {stats['val_images']} 张图片")

    # 生成dataset.yaml
    yaml_path = os.path.join(yolo_output, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: .\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['TBJU_region']\n")
    print(f"    生成: dataset.yaml")

    # 生成OCR数据集
    print(f"\n  生成OCR识别数据集...")
    ocr_output = os.path.join(output_root, f'wagon_number_ocr_{dataset_name}')

    train_ocr = os.path.join(ocr_output, 'train')
    train_crops, train_labels, train_review = generate_ocr_dataset(
        train_annotations, os.path.join(dataset_dir, 'train', 'images'),
        train_ocr, 'train'
    )
    stats['train_crops'] = train_crops
    print(f"    Train crops: {train_crops}, 有效: {train_labels}, review: {train_review}")

    val_ocr = os.path.join(ocr_output, 'val')
    val_crops, val_labels, val_review = generate_ocr_dataset(
        val_annotations, os.path.join(dataset_dir, 'val', 'images'),
        val_ocr, 'val'
    )
    stats['val_crops'] = val_crops
    print(f"    Val crops: {val_crops}, 有效: {val_labels}, review: {val_review}")

    # 合并CSV
    merge_csvs(ocr_output, 'train', 'val')
    merge_review_csvs(ocr_output, 'train', 'val')

    return stats


def merge_csvs(output_dir, *splits):
    """合并多个split的labels.csv"""
    all_rows = []
    for split in splits:
        csv_path = os.path.join(output_dir, split, 'labels.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)

    if all_rows:
        merged_path = os.path.join(output_dir, 'labels.csv')
        with open(merged_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['crop_name', 'original_image', 'text', 'type', 'x1', 'y1', 'x2', 'y2', 'split'])
            writer.writeheader()
            writer.writerows(all_rows)


def merge_review_csvs(output_dir, *splits):
    """合并多个split的review_list.csv"""
    all_rows = []
    for split in splits:
        csv_path = os.path.join(output_dir, split, 'review_list.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)

    if all_rows:
        merged_path = os.path.join(output_dir, 'review_list.csv')
        with open(merged_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['original_image', 'x1', 'y1', 'x2', 'y2', 'text', 'reason'])
            writer.writeheader()
            writer.writerows(all_rows)


def save_summary(output_dir, stats_list):
    """保存汇总统计"""
    summary_path = os.path.join(output_dir, 'summary.txt')

    total = {k: 0 for k in stats_list[0].keys()} if stats_list else {}
    for stats in stats_list:
        for key in total:
            total[key] += stats.get(key, 0)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("数据集转换统计汇总\n")
        f.write("="*60 + "\n\n")
        f.write(f"总Task数: {total.get('total_tasks', 0)}\n")
        f.write(f"成功匹配图片数: {total.get('matched_images', 0)}\n")
        f.write(f"找不到图片数: {total.get('unmatched_images', 0)}\n")
        f.write(f"总标注框数: {total.get('total_boxes', 0)}\n")
        f.write(f"有效TBJU/TBCU标注数: {total.get('valid_tbju_boxes', 0)}\n")
        f.write(f"Review数量: {total.get('review_count', 0)}\n")
        f.write(f"Train图片数: {total.get('train_images', 0)}\n")
        f.write(f"Val图片数: {total.get('val_images', 0)}\n")
        f.write(f"Train crop数: {total.get('train_crops', 0)}\n")
        f.write(f"Val crop数: {total.get('val_crops', 0)}\n")

    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    for key, value in total.items():
        print(f"{key}: {value}")
    print(f"\n统计已保存到: {summary_path}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description='Label Studio JSON → YOLO + OCR 数据集转换')
    parser.add_argument('--dataset_dirs', nargs='+', required=True,
                        help='数据集目录列表，如: ./平视 ./斜视')
    parser.add_argument('--output', required=True,
                        help='输出根目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')

    args = parser.parse_args()

    print("="*60)
    print("Label Studio JSON → YOLO + OCR 数据集转换脚本")
    print("="*60)
    print(f"数据集目录: {args.dataset_dirs}")
    print(f"输出目录: {args.output}")
    print(f"随机种子: {args.seed}")

    os.makedirs(args.output, exist_ok=True)

    all_stats = []
    for dataset_dir in args.dataset_dirs:
        if os.path.exists(dataset_dir):
            stats = process_dataset(dataset_dir, args.output, args.seed)
            all_stats.append(stats)
        else:
            print(f"\n[警告] 数据集目录不存在: {dataset_dir}")

    if all_stats:
        save_summary(args.output, all_stats)

    print("\n" + "="*60)
    print("转换完成！")
    print("="*60)


if __name__ == '__main__':
    main()
