# OCR 模型输出目录

存放 OCR 识别模型的训练产物。

---

## 目录树

```
output/
├── README.md                              # 本文件
│
└── ppocr_rec_carriage_number/             # 车号 OCR 模型
    ├── best_model.pth                     #   最佳模型权重
    ├── latest_model.pth                   #   最新 checkpoint
    ├── ppocr_rec_tbju.onnx                #   ONNX 模型（转 RKNN 用）
    ├── ppocr_keys_v1.txt                  #   字符字典副本
    ├── training_log.json                  #   训练日志（JSON）
    ├── evaluation_results.csv             #   每 epoch 评估指标
    └── 训练日志.txt                        #   训练日志（文本）
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `best_model.pth` | 验证集准确率最高的模型，用于测试和推理 |
| `ppocr_rec_tbju.onnx` | ONNX 格式，用 rknn-toolkit2 转 RKNN 部署到 RK3588 |
| `training_log.json` | 每 epoch 的 loss 和准确率（可画训练曲线） |
| `evaluation_results.csv` | 每 epoch 的评估指标（准确率、各视角统计） |
| `ppocr_keys_v1.txt` | 字符字典：`0123456789BCJTU`（15 字符 + CTC blank） |

## 模型架构

```
输入 (3×32×W) → MobileNetV3-Small → BiLSTM×2 → FC → CTC Decode → "TBJU6970527"
```

## 部署路径

```
best_model.pth → export → ppocr_rec_tbju.onnx → rknn-toolkit2 → model.rknn → RK3588
```
