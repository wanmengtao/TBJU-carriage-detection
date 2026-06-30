"""
PP-OCR Rec 共享模型定义
MobileNetV3-Small + BiLSTM + CTC
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================
# 字符字典
# ============================================================

DEFAULT_CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                 "B", "C", "J", "T", "U"]


def load_char_dict(dict_path: str) -> List[str]:
    """加载字符字典文件"""
    chars = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chars.append(line)
    return chars


# ============================================================
# 模型定义
# ============================================================

class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small backbone，输出特征图"""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hswish = nn.Hardswish(inplace=True)

        self.blocks = nn.ModuleList([
            self._make_block(16, 16, 3, 2, 1, "relu"),
            self._make_block(16, 24, 3, 2, 1, "relu"),
            self._make_block(24, 24, 3, 1, 1, "relu"),
            self._make_block(24, 40, 5, 2, 2, "hardswish"),
            self._make_block(40, 40, 5, 1, 2, "hardswish"),
            self._make_block(40, 40, 5, 1, 2, "hardswish"),
            self._make_block(40, 48, 5, 1, 2, "hardswish"),
            self._make_block(48, 48, 5, 1, 2, "hardswish"),
            self._make_block(48, 96, 5, 2, 2, "hardswish"),
            self._make_block(96, 96, 5, 1, 2, "hardswish"),
            self._make_block(96, 96, 5, 1, 2, "hardswish"),
        ])

        self.conv_last = nn.Conv2d(96, 576, 1, bias=False)
        self.bn_last = nn.BatchNorm2d(576)
        self.conv_out = nn.Conv2d(576, 960, 1, bias=False)
        self.bn_out = nn.BatchNorm2d(960)

    def _make_block(self, in_ch, out_ch, kernel, stride, padding, activation):
        layers = []
        layers.append(nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False))
        layers.append(nn.BatchNorm2d(in_ch))
        layers.append(nn.Hardswish(inplace=True) if activation == "hardswish" else nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.Hardswish(inplace=True) if activation == "hardswish" else nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.hswish(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        x = self.hswish(self.bn_last(self.conv_last(x)))
        x = self.hswish(self.bn_out(self.conv_out(x)))
        return x


class PPoCRRec(nn.Module):
    """
    PP-OCR Rec 识别模型
    架构: MobileNetV3-Small -> Reshape -> BiLSTM -> FC -> CTC
    """

    def __init__(
        self,
        num_classes: int = 16,
        in_channels: int = 3,
        hidden_size: int = 96,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV3Small(in_channels)
        self.rnn = nn.LSTM(
            input_size=960,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 32, W) 输入图像
        Returns:
            logits: (B, T, num_classes) 序列预测
        """
        feat = self.backbone(x)
        # 高度维度应为 1，用 mean 更安全
        feat = feat.mean(dim=2)
        feat = feat.permute(0, 2, 1)
        rnn_out, _ = self.rnn(feat)
        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)
        return logits


# ============================================================
# CTC 解码
# ============================================================

def ctc_decode(
    logits: torch.Tensor,
    chars: List[str],
    blank_idx: int = 0,
) -> List[str]:
    """
    CTC greedy decode
    Args:
        logits: (B, T, num_classes) 或 (T, num_classes)
        chars: 字符列表（不含 blank）
        blank_idx: blank 索引
    Returns:
        解码后的字符串列表
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    preds = logits.argmax(dim=-1)

    results = []
    for pred in preds:
        chars_list = []
        prev = None
        for idx in pred:
            idx = idx.item()
            if idx != blank_idx and idx != prev and 1 <= idx <= len(chars):
                chars_list.append(chars[idx - 1])
            prev = idx
        results.append("".join(chars_list))

    return results


# ============================================================
# 图像预处理
# ============================================================

def preprocess_ocr_image(image: Image.Image, max_width: int = 384) -> torch.Tensor:
    """预处理图像为 PP-OCR Rec 输入格式: (3, 32, W), normalize to [-1, 1]"""
    w, h = image.size
    new_h = 32
    new_w = max(1, int(w * new_h / h))
    # 确保宽度在 [384, max_width] 范围内，保证 T >= 12
    min_width = 384
    new_w = max(min_width, min(new_w, max_width))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np)


# ============================================================
# 模型加载
# ============================================================

def load_ppocr_model(model_path: str, device: str = "cuda"):
    """
    加载 PP-OCR Rec 模型
    Returns:
        (model, chars) 元组
    Raises:
        FileNotFoundError: 找不到字符字典文件
    """
    model_dir = Path(model_path).parent
    dict_path = model_dir / "ppocr_keys_v1.txt"
    if not dict_path.exists():
        parent_dir = Path(__file__).parent.parent
        dict_path = parent_dir / "OCR_train" / "ppocr_keys_v1.txt"

    if not dict_path.exists():
        raise FileNotFoundError(
            f"找不到字符字典 ppocr_keys_v1.txt，"
            f"请检查以下路径: {model_dir} 或 {Path(__file__).parent}"
        )

    chars = load_char_dict(str(dict_path))
    if not chars:
        raise ValueError(f"字符字典为空: {dict_path}")

    num_classes = len(chars) + 1
    model = PPoCRRec(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    return model, chars


def ocr_recognize(model, chars, image: Image.Image, device: str = "cuda") -> str:
    """单张图片 OCR 识别"""
    tensor = preprocess_ocr_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        log_probs = F.log_softmax(logits, dim=-1)
    result = ctc_decode(log_probs, chars)
    return result[0]
