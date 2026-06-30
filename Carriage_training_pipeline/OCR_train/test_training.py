"""
快速验证训练流程
用合成数据测试模型能否学习，不需要真实数据集和 GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ocr_model import PPoCRRec, ctc_decode, DEFAULT_CHARS


class SyntheticDataset(Dataset):
    """合成数据集：生成固定的图像和标签"""

    def __init__(self, chars, num_samples=100):
        self.chars = chars
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.num_samples = num_samples
        self.img_height = 32
        self.img_width = 384

        # 固定的标签
        self.labels = ["TBJU6970527", "TBJU4950882", "TBJU0633534"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机图像
        image = torch.randn(3, self.img_height, self.img_width)

        # 循环使用标签
        text = self.labels[idx % len(self.labels)]
        label = [self.char_to_idx[c] for c in text]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "label_length": len(label),
            "text": text,
            "crop_name": f"synthetic_{idx}.jpg",
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = [item["label"] for item in batch]
    label_lengths = torch.tensor([item["label_length"] for item in batch], dtype=torch.long)
    texts = [item["text"] for item in batch]

    max_label_len = max(label_lengths)
    padded_labels = []
    for label in labels:
        if len(label) < max_label_len:
            pad = torch.zeros(max_label_len - len(label), dtype=torch.long)
            label = torch.cat([label, pad])
        padded_labels.append(label)

    return {
        "images": images,
        "labels": torch.stack(padded_labels),
        "label_lengths": label_lengths,
        "texts": texts,
    }


def test_training():
    print("=" * 60)
    print("快速验证训练流程（合成数据，CPU）")
    print("=" * 60)

    chars = DEFAULT_CHARS
    num_classes = len(chars) + 1
    device = "cpu"

    # 创建模型
    model = PPoCRRec(num_classes=num_classes)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 创建数据集
    train_dataset = SyntheticDataset(chars, num_samples=100)
    val_dataset = SyntheticDataset(chars, num_samples=30)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # 训练 10 个 epoch
    print("\n开始训练...")
    for epoch in range(10):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)

            seq_len = log_probs.shape[0]
            input_lengths = torch.full((images.shape[0],), seq_len, dtype=torch.long, device=device)

            loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=-1)
                predictions = ctc_decode(log_probs, chars)

                for pred, gt in zip(predictions, batch["texts"]):
                    val_total += 1
                    if pred.lower() == gt.lower():
                        val_correct += 1

        val_accuracy = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:2d}/10 - Loss: {train_loss:.4f} - 准确率: {val_accuracy:.4f} ({val_correct}/{val_total})")

    print("\n" + "=" * 60)
    if val_accuracy > 0:
        print("验证通过！模型可以学习，训练流程正确。")
        print("可以搬到 GPU 服务器上进行正式训练。")
    else:
        print("验证失败！模型未能学习，请检查代码。")
    print("=" * 60)


if __name__ == "__main__":
    test_training()
