# 模型文件目录

本目录包含 RK3588 推理所需的预训练模型文件（RKNN 格式）。

---

## 目录结构

```
models/
├── yolo/                              # YOLO 检测模型
│   ├── merged_yolov8.rknn            # 默认模型 (FP16, ~23MB)
│   ├── merged_yolov8_fp.rknn         # FP16 备份 (~23MB)
│   └── merged_yolov8_i8.rknn         # INT8 量化 (~12MB, 速度优先)
│
└── ocr/                               # OCR 识别模型
    ├── rec_tbju.rknn                  # 默认模型 (~4MB)
    └── rec_tbju_fp.rknn               # FP16 备份 (~4MB)
```

---

## 模型说明

### YOLO 检测模型

| 模型 | 精度 | 大小 | 说明 |
|------|------|------|------|
| `merged_yolov8.rknn` | FP16 | ~23MB | 默认使用，平衡精度和速度 |
| `merged_yolov8_fp.rknn` | FP16 | ~23MB | 备份模型 |
| `merged_yolov8_i8.rknn` | INT8 | ~12MB | 速度优先，精度略低 |

**检测类别（6类）：**
| class_id | 类别 | 说明 |
|----------|------|------|
| 0 | TBJU_region | 车号区域 |
| 1 | carriage_rim_region | 车厢沿区域 |
| 2 | carriage_rim_debris | 车厢沿异物 |
| 3 | track_region | 轨道区域 |
| 4 | track_intrusion_debris | 轨道异物 |
| 5 | door_region | 车门区域 |

### OCR 识别模型

| 模型 | 大小 | 说明 |
|------|------|------|
| `rec_tbju.rknn` | ~4MB | 默认使用 |
| `rec_tbju_fp.rknn` | ~4MB | 备份模型 |

**识别字符（15个）：**
- 数字：0-9
- 字母：B, C, J, T, U
- 输出格式：TBJU + 7位数字（如 TBJU4950882）

---

## 模型获取

RKNN 模型已包含在本仓库中。

如需重新训练模型，请参考：
- YOLO 模型训练：[Carriage_training_pipeline/YOLO_train/](../../Carriage_training_pipeline/YOLO_train/)
- OCR 模型训练：[Carriage_training_pipeline/OCR_train/](../../Carriage_training_pipeline/OCR_train/)

---

## 使用方式

模型文件在 `run_gui.py` 启动时自动加载，无需手动指定路径。

如需切换模型，在 GUI 的"模型切换"下拉框中选择。
