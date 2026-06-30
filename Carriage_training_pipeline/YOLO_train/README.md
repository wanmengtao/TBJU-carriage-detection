# YOLO 检测模型训练

## 功能说明

使用 ultralytics YOLOv8 进行目标检测训练，支持 RKNN 格式导出（RK3588 部署）。

包含三类训练脚本：

| 脚本 | 任务 | 类别 |
|------|------|------|
| `train_yolo_wagon_number.py` | 车号区域检测 | TBJU_region (nc=1) |
| `train_yolo_debris.py` | 异物/车门检测 | 异物: region+debris (nc=2), 车门: region (nc=1) |
| `train_yolo_merged.py` | 统一模型（合并四任务） | 6类 (nc=6) |

## 安装依赖

```bash
pip install ultralytics opencv-python pyyaml
```

---

## 车号检测训练

```bash
# 使用配置文件（推荐）
python YOLO_train/train_yolo_wagon_number.py --use_config

# 直接指定路径
python YOLO_train/train_yolo_wagon_number.py --dataset_dir datasets/output

# 含 RKNN 导出
python YOLO_train/train_yolo_wagon_number.py --use_config --export_rknn
```

会自动合并平视+侧视数据集，生成 `wagon_number_detection_merged/`。

---

## 异物/车门检测训练

三个任务共用同一个脚本，通过 `--task` 参数区分：

```bash
# 车厢沿异物
python YOLO_train/train_yolo_debris.py --task carriage_rim_debris --use_config

# 轨道异物侵限
python YOLO_train/train_yolo_debris.py --task track_intrusion --use_config

# 车门状态
python YOLO_train/train_yolo_debris.py --task door_state --use_config

# 跳过 ONNX 导出
python YOLO_train/train_yolo_debris.py --task track_intrusion --use_config --no_export
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --task | 任务名 (carriage_rim_debris / track_intrusion / door_state) | 必填 |
| --use_config | 使用 config.py 中的路径配置 | False |
| --dataset_dir | 数据集目录（不用 config 时必填） | — |
| --model_size | 模型大小 (n/s/m/l/x) | s |
| --epochs | 训练轮数 | 100 |
| --batch_size | 批量大小 | 16 |
| --img_size | 输入图片尺寸 | 640 |
| --device | 训练设备 (0=GPU0, cpu=CPU) | 0 |
| --no_export | 跳过 ONNX 导出 | False |

---

## 统一模型训练（合并四任务）

将车号检测、车厢沿异物、轨道异物侵限、车门状态四个任务合并为一个模型（nc=6）：

```bash
python YOLO_train/train_yolo_merged.py --use_config
python YOLO_train/train_yolo_merged.py --use_config --epochs 150 --batch_size 8
```

统一类别：
| class_id | 类别 |
|----------|------|
| 0 | TBJU_region |
| 1 | carriage_rim_region |
| 2 | carriage_rim_debris |
| 3 | track_region |
| 4 | track_intrusion_debris |
| 5 | door_region |

输出目录：`YOLO_train/run_merged/train/weights/`

---

## 输出说明

训练产物保存在 `YOLO_train/runs/{task}_detection/` 或 `YOLO_train/run_merged/`：

```
runs/{task}_detection/
├── weights/
│   ├── best.pt          # 最佳模型（PyTorch）
│   ├── best.onnx        # ONNX 模型（用于转 RKNN）
│   └── last.pt          # 最新模型
├── results.png          # 训练曲线
├── confusion_matrix.png # 混淆矩阵
└── ...
```

训练完成后自动导出 ONNX 模型，路径与 best.pt 同目录。使用 `--no_export` 可跳过导出。

---

## 模型选择建议

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| yolov8n | Nano，最轻量 | 边缘设备，精度要求不高 |
| yolov8s | Small，推荐 | 平衡性能和精度 |
| yolov8m | Medium | 精度要求较高 |
| yolov8l | Large | 高精度，训练时间较长 |
| yolov8x | XLarge | 最高精度，训练时间最长 |

---

## ONNX → RKNN 部署

训练完成后自动导出 ONNX 模型。使用 rknn-toolkit2 转换为 RKNN 格式：

```bash
pip install rknn-toolkit2
```

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
rknn.load_onnx(model='runs/{task}_detection/weights/best.onnx')
rknn.build(do_quantization=True, dataset='calibration_list.txt')
rknn.export_rknn(f'{task}_detection.rknn')
```
