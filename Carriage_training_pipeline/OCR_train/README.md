# PP-OCR Rec 车号识别模型训练

## 功能说明

- 合并平视和侧视 OCR 数据集进行联合训练
- 基于 PP-OCR Rec 架构 (MobileNetV3 + BiLSTM + CTC) 进行车号文本识别
- 支持 ONNX 格式导出（用于 RK3588 部署）
- 轻量模型，适合嵌入式部署

## 安装依赖

```bash
pip install torch torchvision pillow numpy
```

## 使用方法

### 方法一：使用配置文件（推荐）

config.py 会自动推导项目根目录，无需手动修改路径。

2. 运行训练：

```bash
python train_ocr.py --use_config
```

### 方法二：直接指定路径

```bash
python train_ocr.py --dataset_dir datasets/output
```

### 完整训练（含 ONNX 导出）

```bash
python train_ocr.py --use_config --epochs 50 --batch_size 16 --learning_rate 1e-3 --export_onnx
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --use_config | 使用 config.py 中的配置 | False |
| --dataset_dir | 数据集目录（如果使用 --use_config 则无需指定） | 必填（不用配置文件时） |
| --char_dict | 字符字典文件路径 | ppocr_keys_v1.txt |
| --epochs | 训练轮数 | 50 |
| --batch_size | 批量大小 | 16 |
| --learning_rate | 学习率 | 1e-3 |
| --max_width | 图像最大宽度 | 320 |
| --device | 训练设备 (cuda/cpu) | cuda |
| --use_amp | 使用混合精度训练 | True |
| --no_augment | 禁用数据增强 | False |
| --export_onnx | 导出 ONNX 格式 | False |
| --skip_merge | 跳过数据集合并 | False |
| --merged_dir | 已合并的数据集目录 | None |
| --output_dir | 模型输出目录 | OCR_train/output |

## 模型架构

```
输入图像 (3×32×W)
    ↓
MobileNetV3-Small (backbone) → (960×1×W/4)
    ↓
Reshape → (W/4, 960) 序列
    ↓
BiLSTM×2 (hidden=96) → (W/4, 192)
    ↓
FC (192 → num_classes) → (W/4, 16)  # 15字符 + 1 blank
    ↓
CTC Decode → "TBJU6970527"
```

字符集: `0123456789BCJTU` (15字符) + CTC blank = 16 类

## 输出说明

训练完成后会在以下位置生成文件：

```
wagon_number_ocr_merged/
├── train/
│   ├── crops/           # 训练 crop 图片
│   └── labels.csv       # 训练标签
├── val/
│   ├── crops/           # 验证 crop 图片
│   └── labels.csv       # 验证标签
└── ...

ppocr_rec_carriage_number/
├── best_model.pth       # 最佳模型
├── latest_model.pth     # 最新模型
├── ppocr_keys_v1.txt    # 字符字典
├── training_log.json    # 训练日志
├── evaluation_results.csv # 评估结果
└── ppocr_rec_tbju.onnx  # ONNX 模型（如果使用 --export_onnx）
```

## 注意事项

1. **GPU 要求**: 训练需要 GPU，建议显存 >= 4GB（PP-OCR Rec 比 TrOCR 轻量很多）
2. **训练时间**: 50 epochs 约需 30 分钟 - 1 小时（取决于 GPU 性能）
3. **数据增强**: 默认启用，可提高模型泛化能力
4. **混合精度**: 默认启用，可加速训练并节省显存

## RKNN 部署说明

导出的 ONNX 模型使用 rknn-toolkit2 转换为 RKNN 格式：

```bash
# 在 Ubuntu 虚拟机中
conda activate RKNN-Toolkit2-2.3.2

# 使用 RKNN Model Zoo 的 PPOCR 例程转换
cd ~/rknn_model_zoo-2.3.2/examples/ppocr/python
```

PP-OCR Rec 的 ONNX 导出结构简单（固定输入高度 32，宽度最大 320），适合 RKNN 转换。
