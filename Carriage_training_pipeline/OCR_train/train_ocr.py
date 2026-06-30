"""
OCR 识别模型训练脚本
基于 PP-OCR Rec 架构 (MobileNetV3 + BiLSTM + CTC) 训练 TBJU 车号文本识别模型
适合嵌入式部署 (RK3588 / RKNN)
"""

import os
import sys
import csv
import json
import shutil
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image



# 从共享模块导入模型定义
from ocr_model import (
    MobileNetV3Small, PPoCRRec, ctc_decode,
    load_char_dict, DEFAULT_CHARS,
)


# ============================================================
# 数据集
# ============================================================

class PPoCRRecDataset(Dataset):
    """
    PP-OCR Rec 数据集
    读取 labels.csv，预处理图像为 (3, 32, W) 格式
    """

    def __init__(
        self,
        crops_dir: str,
        labels_csv: str,
        chars: List[str],
        max_width: int = 384,
        augment: bool = False,
    ):
        self.crops_dir = Path(crops_dir)
        self.chars = chars
        self.max_width = max_width
        self.augment = augment
        self.img_height = 32
        # 最小宽度 = (最大标签长度 + margin) * backbone步长
        # backbone 步长 = 32, 标签长度 11, margin 1 → (11+1)*32 = 384
        self.min_width = 384

        # 字符到索引的映射 (0 保留给 blank)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}

        # 读取 labels.csv
        self.samples = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                crop_path = self.crops_dir / row["crop_name"]
                if crop_path.exists():
                    self.samples.append({
                        "image_path": str(crop_path),
                        "text": row["text"],
                        "crop_name": row["crop_name"],
                    })

        print(f"加载 {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 加载并预处理图像
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self._preprocess(image)

        # 编码文本
        text = sample["text"]
        label = []
        for c in text:
            if c in self.char_to_idx:
                label.append(self.char_to_idx[c])
            else:
                warnings.warn(f"字符集外字符 '{c}' in '{text}' ({sample['crop_name']}), 已跳过")

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "label_length": len(label),
            "text": text,
            "crop_name": sample["crop_name"],
        }

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """预处理图像: resize to 32xW, normalize to [-1, 1]"""
        w, h = image.size
        new_h = self.img_height
        new_w = max(1, int(w * new_h / h))
        # 确保宽度在 [min_width, max_width] 范围内
        new_w = max(self.min_width, min(new_w, self.max_width))

        image = image.resize((new_w, new_h), Image.BILINEAR)

        if self.augment:
            image = self._augment(image)

        img_np = np.array(image, dtype=np.float32) / 255.0
        img_np = (img_np - 0.5) / 0.5
        img_np = img_np.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_np)

        # RandomErasing 需要 tensor 输入
        if self.augment:
            tensor = self._augment_tensor(tensor)

        return tensor

    def _augment(self, image: Image.Image) -> Image.Image:
        """数据增强（PIL Image 阶段）：模拟真实场景中的各种干扰"""
        import random as _random

        # 随机透视变换（模拟不同拍摄角度）
        if _random.random() < 0.3:
            w, h = image.size
            magnitude = 0.05
            dx = int(w * magnitude)
            dy = int(h * magnitude)
            src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
            dst_pts = [
                (_random.randint(-dx, dx), _random.randint(-dy, dy)),
                (w + _random.randint(-dx, dx), _random.randint(-dy, dy)),
                (w + _random.randint(-dx, dx), h + _random.randint(-dy, dy)),
                (_random.randint(-dx, dx), h + _random.randint(-dy, dy)),
            ]
            from PIL import ImageTransform
            image = image.transform((w, h), ImageTransform.QuadTransform(
                [coord for pt in dst_pts for coord in pt]
            ), fillcolor=128)

        # 随机旋转
        if _random.random() < 0.5:
            angle = _random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor=128, expand=False)

        # 随机仿射
        if _random.random() < 0.5:
            from torchvision import transforms
            affine = transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=3,
                fill=128,
            )
            image = affine(image)

        # 颜色扰动
        if _random.random() < 0.7:
            from torchvision import transforms
            color_jitter = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.1,
                hue=0.02,
            )
            image = color_jitter(image)

        # 随机模糊
        if _random.random() < 0.3:
            from torchvision import transforms
            blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
            image = blur(image)

        # 随机噪声
        if _random.random() < 0.2:
            image = self._add_noise(image)

        return image

    def _augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """数据增强（tensor 阶段）：随机遮挡"""
        from torchvision import transforms
        erasing = transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
        return erasing(tensor)

    @staticmethod
    def _add_noise(image: Image.Image) -> Image.Image:
        """添加高斯噪声"""
        import random
        img_np = np.array(image, dtype=np.float32)
        noise_level = random.uniform(5, 20)
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class PPoCRRecCollator:
    """动态宽度 padding 的 collator"""

    def __init__(self, max_width: int = 384):
        self.max_width = max_width

    def __call__(self, batch: List[Dict]) -> Dict:
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        label_lengths = [item["label_length"] for item in batch]
        texts = [item["text"] for item in batch]
        crop_names = [item["crop_name"] for item in batch]

        # 找到 batch 中最大宽度
        max_w = max(img.shape[2] for img in images)
        max_w = min(max_w, self.max_width)

        # Pad images to same width
        padded_images = []
        for img in images:
            c, h, w = img.shape
            if w < max_w:
                pad = torch.zeros(c, h, max_w - w)
                img = torch.cat([img, pad], dim=2)
            elif w > max_w:
                img = img[:, :, :max_w]
            padded_images.append(img)

        # Pad labels to same length
        max_label_len = max(label_lengths) if label_lengths else 1
        padded_labels = []
        for label in labels:
            if len(label) < max_label_len:
                pad = torch.zeros(max_label_len - len(label), dtype=torch.long)
                label = torch.cat([label, pad])
            padded_labels.append(label)

        return {
            "images": torch.stack(padded_images),
            "labels": torch.stack(padded_labels),
            "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
            "texts": texts,
            "crop_names": crop_names,
        }


# ============================================================
# 数据集合并
# ============================================================

def merge_ocr_datasets(output_dir: str) -> Tuple[str, str, str, str]:
    """
    合并平视和侧视 OCR 数据集
    返回: (merged_crops_train, merged_crops_val, merged_labels_train, merged_labels_val)
    """
    print("=" * 60)
    print("步骤 1: 合并平视和侧视 OCR 数据集")
    print("=" * 60)

    eye_level_dir = Path(output_dir) / "wagon_number_ocr_平视"
    side_level_dir = Path(output_dir) / "wagon_number_ocr_侧视"
    merged_dir = Path(output_dir) / "wagon_number_ocr_merged"

    for split in ["train", "val"]:
        (merged_dir / split / "crops").mkdir(parents=True, exist_ok=True)

    # 合并 train 数据
    all_train_labels = []
    for dataset_dir in [eye_level_dir, side_level_dir]:
        train_csv = dataset_dir / "train" / "labels.csv"
        if train_csv.exists():
            with open(train_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src_crop = dataset_dir / "train" / "crops" / row["crop_name"]
                    dst_crop = merged_dir / "train" / "crops" / row["crop_name"]
                    if src_crop.exists() and not dst_crop.exists():
                        shutil.copy2(src_crop, dst_crop)
                    all_train_labels.append(row)

    # 合并 val 数据
    all_val_labels = []
    for dataset_dir in [eye_level_dir, side_level_dir]:
        val_csv = dataset_dir / "val" / "labels.csv"
        if val_csv.exists():
            with open(val_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src_crop = dataset_dir / "val" / "crops" / row["crop_name"]
                    dst_crop = merged_dir / "val" / "crops" / row["crop_name"]
                    if src_crop.exists() and not dst_crop.exists():
                        shutil.copy2(src_crop, dst_crop)
                    all_val_labels.append(row)

    # 保存合并后的 labels.csv
    csv_columns = ["crop_name", "original_image", "text", "type", "x1", "y1", "x2", "y2", "split"]

    train_csv_path = merged_dir / "train" / "labels.csv"
    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_train_labels)

    val_csv_path = merged_dir / "val" / "labels.csv"
    with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_val_labels)

    # 统计
    train_crops = len(list((merged_dir / "train" / "crops").iterdir()))
    val_crops = len(list((merged_dir / "val" / "crops").iterdir()))

    print(f"\n合并完成: {merged_dir}")
    print(f"  Train: {train_crops} 张 crop 图片, {len(all_train_labels)} 条标签")
    print(f"  Val: {val_crops} 张 crop 图片, {len(all_val_labels)} 条标签")

    return (
        str(merged_dir / "train" / "crops"),
        str(merged_dir / "val" / "crops"),
        str(train_csv_path),
        str(val_csv_path),
    )


# ============================================================
# 训练
# ============================================================

def train_ocr(
    train_crops_dir: str,
    train_labels_csv: str,
    val_crops_dir: str,
    val_labels_csv: str,
    output_dir: str,
    chars: List[str],
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    max_width: int = 384,
    device: str = "cuda",
    use_amp: bool = True,
    augment: bool = True,
):
    """训练 PP-OCR Rec 模型"""
    print("\n" + "=" * 60)
    print("步骤 2: 训练 PP-OCR Rec 识别模型")
    print("=" * 60)

    output_path = Path(output_dir) / "ppocr_rec_carriage_number"
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存字符字典到模型目录
    dict_path = output_path / "ppocr_keys_v1.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        for c in chars:
            f.write(c + "\n")

    # 创建模型
    num_classes = len(chars) + 1
    model = PPoCRRec(num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型架构: PP-OCR Rec (MobileNetV3 + BiLSTM + CTC)")
    print(f"字符集大小: {len(chars)} 字符 + 1 blank = {num_classes} 类")
    print(f"字符集: {''.join(chars)}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 创建数据集
    print("\n加载训练数据...")
    train_dataset = PPoCRRecDataset(
        crops_dir=train_crops_dir,
        labels_csv=train_labels_csv,
        chars=chars,
        max_width=max_width,
        augment=augment,
    )

    print("加载验证数据...")
    val_dataset = PPoCRRecDataset(
        crops_dir=val_crops_dir,
        labels_csv=val_labels_csv,
        chars=chars,
        max_width=max_width,
        augment=False,
    )

    # 调试：检查前几个样本的尺寸和标签
    print("\n[调试] 检查前 5 个训练样本:")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        img_shape = sample["image"].shape
        label = sample["label"]
        text = sample["text"]
        print(f"  样本 {i}: image={img_shape}, label={label.tolist()}, text='{text}', label_len={sample['label_length']}")

    collator = PPoCRRecCollator(max_width=max_width)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

    # 带预热的余弦退火调度器
    warmup_epochs = min(5, epochs // 10)
    main_epochs = epochs - warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / main_epochs
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # CTC Loss
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    # 训练循环
    print(f"\n开始训练: {epochs} epochs (预热 {warmup_epochs} 轮)")
    best_accuracy = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()

            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)

                    # 调试：第一个 batch 的形状信息
                    if epoch == 0 and batch_idx == 0:
                        print(f"\n[调试] 第一个 batch:")
                        print(f"  images shape: {images.shape}")
                        print(f"  logits shape: {logits.shape}")
                        print(f"  labels shape: {labels.shape}")
                        print(f"  label_lengths: {label_lengths.tolist()}")
                        print(f"  logits min/max: {logits.min():.4f}/{logits.max():.4f}")
                        print(f"  logits has NaN: {torch.isnan(logits).any()}")
                        print(f"  模型输出时间步 T={logits.shape[1]}, 最大标签长度={label_lengths.max().item()}")

                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs = log_probs.permute(1, 0, 2)

                    seq_len = log_probs.shape[0]
                    max_label_len = label_lengths.max().item()
                    if seq_len < max_label_len:
                        warnings.warn(f"序列长度 T={seq_len} < 最大标签长度 {max_label_len}，CTC Loss 将返回 0。请增大 max_width。")

                    input_lengths = torch.full((images.shape[0],), seq_len, dtype=torch.long, device=device)

                    loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)

                # 调试：第一个 batch 的形状信息
                if epoch == 0 and batch_idx == 0:
                    print(f"\n[调试] 第一个 batch:")
                    print(f"  images shape: {images.shape}")
                    print(f"  logits shape: {logits.shape}")
                    print(f"  labels shape: {labels.shape}")
                    print(f"  label_lengths: {label_lengths.tolist()}")
                    print(f"  logits min/max: {logits.min():.4f}/{logits.max():.4f}")
                    print(f"  logits has NaN: {torch.isnan(logits).any()}")
                    print(f"  模型输出时间步 T={logits.shape[1]}, 最大标签长度={label_lengths.max().item()}")

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.permute(1, 0, 2)

                seq_len = log_probs.shape[0]
                max_label_len = label_lengths.max().item()
                if seq_len < max_label_len:
                    warnings.warn(f"序列长度 T={seq_len} < 最大标签长度 {max_label_len}，CTC Loss 将返回 0。请增大 max_width。")

                input_lengths = torch.full((images.shape[0],), seq_len, dtype=torch.long, device=device)

                loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                labels = batch["labels"].to(device)
                label_lengths = batch["label_lengths"].to(device)

                if use_amp and device == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_t = log_probs.permute(1, 0, 2)
                seq_len = log_probs_t.shape[0]
                input_lengths = torch.full((images.shape[0],), seq_len, dtype=torch.long, device=device)
                loss = ctc_loss_fn(log_probs_t.float(), labels, input_lengths, label_lengths)
                val_loss += loss.item()

                predictions = ctc_decode(log_probs, chars)
                for pred, gt in zip(predictions, batch["texts"]):
                    val_predictions.append({
                        "prediction": pred,
                        "ground_truth": gt,
                    })

        val_loss /= len(val_loader)

        string_correct = sum(
            1 for p in val_predictions
            if p["prediction"].lower() == p["ground_truth"].lower()
        )
        string_accuracy = string_correct / len(val_predictions) if val_predictions else 0

        # 计算字符级准确率
        total_chars = 0
        correct_chars = 0
        for p in val_predictions:
            pred = p["prediction"].lower()
            gt = p["ground_truth"].lower()
            max_len = max(len(pred), len(gt))
            total_chars += max_len
            for j in range(max_len):
                pred_c = pred[j] if j < len(pred) else ""
                gt_c = gt[j] if j < len(gt) else ""
                if pred_c == gt_c:
                    correct_chars += 1
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0

        print(f"\n训练 - Loss: {train_loss:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}")
        print(f"字符串准确率: {string_accuracy:.4f} ({string_correct}/{len(val_predictions)})")
        print(f"字符级准确率: {char_accuracy:.4f} ({correct_chars}/{total_chars})")

        # 调试：打印预测样本
        empty_count = sum(1 for p in val_predictions if not p["prediction"])
        print(f"  [调试] 空预测: {empty_count}/{len(val_predictions)}")
        print(f"  [调试] 前 5 个样本预测 vs 真实:")
        for i, p in enumerate(val_predictions[:5]):
            match = "✓" if p["prediction"].lower() == p["ground_truth"].lower() else "✗"
            print(f"    {match} 预测='{p['prediction']}' 真实='{p['ground_truth']}'")

        # 保存最佳模型（基于准确率，val_loss 作辅助参考）
        if string_accuracy > best_accuracy:
            best_accuracy = string_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_path / "best_model.pth")
            print(f"  * 保存最佳模型 (准确率: {best_accuracy:.4f})")
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停: 验证指标连续 {patience} 轮未改善，在 Epoch {epoch + 1} 停止")
                break

        # 保存最新模型
        torch.save(model.state_dict(), output_path / "latest_model.pth")

        # 保存训练日志
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "string_accuracy": string_accuracy,
        }

        log_file = output_path / "training_log.json"
        logs = []
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    print(f"\n训练完成!")
    print(f"最佳模型: Epoch {best_epoch}, 准确率: {best_accuracy:.4f}")
    print(f"模型保存在: {output_path / 'best_model.pth'}")

    return output_path / "best_model.pth", output_path


# ============================================================
# 评估
# ============================================================

def evaluate_ocr(
    model_path: str,
    val_crops_dir: str,
    val_labels_csv: str,
    output_dir: str,
    chars: List[str],
    max_width: int = 384,
    device: str = "cuda",
):
    """评估 PP-OCR Rec 模型"""
    print("\n" + "=" * 60)
    print("步骤 3: 评估 PP-OCR Rec 模型")
    print("=" * 60)

    # 加载模型
    num_classes = len(chars) + 1
    model = PPoCRRec(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 加载验证数据
    dataset = PPoCRRecDataset(
        crops_dir=val_crops_dir,
        labels_csv=val_labels_csv,
        chars=chars,
        max_width=max_width,
        augment=False,
    )
    collator = PPoCRRecCollator(max_width=max_width)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collator)

    all_predictions = []
    total_correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)

            predictions = ctc_decode(log_probs, chars)

            for pred, gt, crop_name in zip(predictions, batch["texts"], batch["crop_names"]):
                pred_clean = pred.strip().lower()
                gt_clean = gt.strip().lower()

                all_predictions.append({
                    "crop_name": crop_name,
                    "prediction": pred.strip(),
                    "ground_truth": gt,
                    "correct": pred_clean == gt_clean,
                })

                min_len = min(len(pred_clean), len(gt_clean))
                total_correct_chars += sum(1 for j in range(min_len) if pred_clean[j] == gt_clean[j])
                total_chars += max(len(pred_clean), len(gt_clean))

    string_correct = sum(1 for p in all_predictions if p["correct"])
    string_accuracy = string_correct / len(all_predictions) if all_predictions else 0
    char_accuracy = total_correct_chars / total_chars if total_chars > 0 else 0

    print(f"\n评估结果:")
    print(f"  总样本数: {len(all_predictions)}")
    print(f"  字符串准确率: {string_accuracy:.4f} ({string_correct}/{len(all_predictions)})")
    print(f"  字符级准确率: {char_accuracy:.4f}")

    results_path = Path(output_dir) / "evaluation_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_name", "prediction", "ground_truth", "correct"])
        writer.writeheader()
        writer.writerows(all_predictions)

    print(f"详细结果保存在: {results_path}")

    return string_accuracy, char_accuracy


# ============================================================
# ONNX 导出
# ============================================================

def export_ocr_onnx(
    model_path: str,
    output_dir: str,
    chars: List[str],
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 384),
):
    """导出 PP-OCR Rec 模型为 ONNX 格式"""
    print("\n" + "=" * 60)
    print("步骤 4: 导出 ONNX 格式")
    print("=" * 60)

    num_classes = len(chars) + 1
    model = PPoCRRec(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(input_shape)

    onnx_path = Path(output_dir) / "ppocr_rec_tbju.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"ONNX 导出: {onnx_path}")
    print(f"输入尺寸: {input_shape}")
    print(f"输出: (1, T, {num_classes}) logits")

    # 保存字符字典到导出目录
    dict_path = Path(output_dir) / "ppocr_keys_v1.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        for c in chars:
            f.write(c + "\n")
    print(f"字符字典: {dict_path}")

    return onnx_path


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PP-OCR Rec 车号识别模型训练")
    parser.add_argument("--use_config", action="store_true",
                        help="使用 config.py 中的配置（推荐）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="转换后的数据集目录")
    parser.add_argument("--char_dict", type=str, default=None,
                        help="字符字典文件路径（默认使用内置字符集）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--max_width", type=int, default=384, help="图像最大宽度（需 >= 352 以支持 11 字符标签）")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度训练")
    parser.add_argument("--no_augment", action="store_true", help="禁用数据增强")
    parser.add_argument("--skip_merge", action="store_true", help="跳过数据集合并")
    parser.add_argument("--merged_dir", type=str, default=None, help="已合并的数据集目录")
    parser.add_argument("--export_onnx", action="store_true", help="导出 ONNX 格式")
    parser.add_argument("--output_dir", type=str, default=None, help="模型输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("PP-OCR Rec 车号识别模型训练")
    print("=" * 60)

    # 加载字符字典
    if args.char_dict:
        chars = load_char_dict(args.char_dict)
    else:
        default_dict = Path(__file__).parent / "ppocr_keys_v1.txt"
        if default_dict.exists():
            chars = load_char_dict(str(default_dict))
        else:
            chars = DEFAULT_CHARS

    if not chars:
        print("错误: 字符字典为空")
        return

    print(f"字符集: {''.join(chars)} ({len(chars)} 字符)")

    # 使用配置文件
    if args.use_config:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config

        errors = config.validate_paths()
        if errors:
            print("\n路径验证失败:")
            for error in errors:
                print(f"  - {error}")
            print("\n请检查 config.py 中的路径配置")
            return

        config.print_config()
        dataset_dir = config.OUTPUT_DIR
        model_output_dir = args.output_dir or config.OCR_OUTPUT_DIR
        print(f"\n使用配置文件中的路径")
    else:
        if args.dataset_dir is None:
            print("错误: 请指定 --dataset_dir 或使用 --use_config")
            parser.print_help()
            return
        dataset_dir = args.dataset_dir
        model_output_dir = args.output_dir or dataset_dir

    # 步骤 1: 合并数据集
    if args.skip_merge and args.merged_dir:
        merged_dir = args.merged_dir
        train_crops = os.path.join(merged_dir, "train", "crops")
        val_crops = os.path.join(merged_dir, "val", "crops")
        train_labels = os.path.join(merged_dir, "train", "labels.csv")
        val_labels = os.path.join(merged_dir, "val", "labels.csv")
        print(f"使用已合并的数据集: {merged_dir}")
    else:
        train_crops, val_crops, train_labels, val_labels = merge_ocr_datasets(dataset_dir)
        merged_dir = str(Path(dataset_dir) / "wagon_number_ocr_merged")

    # 步骤 2: 训练模型
    model_path, output_path = train_ocr(
        train_crops_dir=train_crops,
        train_labels_csv=train_labels,
        val_crops_dir=val_crops,
        val_labels_csv=val_labels,
        output_dir=model_output_dir,
        chars=chars,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_width=args.max_width,
        device=args.device,
        use_amp=args.use_amp,
        augment=not args.no_augment,
    )

    # 步骤 3: 评估模型
    string_acc, char_acc = evaluate_ocr(
        model_path=str(model_path),
        val_crops_dir=val_crops,
        val_labels_csv=val_labels,
        output_dir=str(output_path),
        chars=chars,
        max_width=args.max_width,
        device=args.device,
    )

    # 步骤 4: 导出 ONNX（可选）
    if args.export_onnx:
        export_ocr_onnx(
            model_path=str(model_path),
            output_dir=str(output_path),
            chars=chars,
        )

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳模型: {model_path}")
    print(f"字符串准确率: {string_acc:.4f}")
    print(f"字符级准确率: {char_acc:.4f}")


if __name__ == "__main__":
    main()
