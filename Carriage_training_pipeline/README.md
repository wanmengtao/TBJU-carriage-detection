# Carriage_training_pipeline — 列车车厢检测模型训练管道

## 1. 项目概述

基于深度学习的列车车厢视觉检测系统**训练管道**，负责数据处理、模型训练与测试验证。

部署目标为 RK3588（ELF2）开发板，推理应用见 [TBJU_edge_inference_app](../TBJU_edge_inference_app/)。

**包含 5 个子任务 + 1 个统一模型：**

| 任务 | 模型 | 类别 | 说明 |
|------|------|------|------|
| 车号检测 | YOLOv8 (nc=1) | TBJU_region | 定位车号区域 |
| 车号识别 | PP-OCR Rec (CRNN+CTC) | — | 识别车号文本（TBJU + 7位数字） |
| 车厢沿异物 | YOLOv8 (nc=2) | region, debris | 检测车厢沿上的异物 |
| 轨道异物侵限 | YOLOv8 (nc=2) | region, debris | 检测轨道区域内的异物 |
| 车门状态 | YOLOv8 (nc=1) | region | 检测车门区域（仅区域定位，无异物检测） |
| 统一模型 | YOLOv8 (nc=6) | 6类 | 合并以上四任务，一次推理完成所有检测 |

**技术栈：** Python, ultralytics YOLOv8, PyTorch, Label Studio, ONNX, rknn-toolkit2

---

## ⚠️ 数据集说明

**数据集文件较大（~3.3GB），不包含在本仓库中。**

| 目录 | 大小 | 说明 |
|------|------|------|
| `raw_data/` | ~1.1GB | 原始数据（图片 + Label Studio 标注 JSON） |
| `datasets/` | ~2.2GB | 转换后的数据集（YOLO 训练用） |

**如需获取数据集，请联系：** [待添加邮箱]

---

## 2. 目录结构

```
Carriage_training_pipeline/
├── config.py                              # 全局路径配置（自动推导，无需修改）
├── requirements.txt                       # Python 依赖
├── README.md                              # 本文件
│
├── scripts/                               # 数据处理脚本
│   ├── convert_labelstudio_tbju.py        #   车号 JSON → YOLO + OCR 数据集
│   ├── convert_labelstudio_new.py         #   异物/车门 JSON → YOLO txt
│   ├── augment_debris.py                  #   异物贴图扩充（生成 _aug 图 + JSON）
│   ├── augment_video_debris.py            #   视频帧异物扩充
│   ├── setup_new_datasets.py              #   同步图片到 datasets/（含扩充图）
│   └── merge_datasets.py                  #   合并四任务数据集为统一训练集（nc=6）
│
├── YOLO_train/                            # YOLO 训练脚本
│   ├── train_yolo_wagon_number.py         #   车号检测训练（自动合并平视+侧视）
│   ├── train_yolo_debris.py               #   异物/车门检测训练（--task 参数选择任务）
│   └── train_yolo_merged.py               #   统一模型训练（nc=6，合并四任务）
│
├── OCR_train/                             # OCR 训练脚本
│   ├── train_ocr.py                       #   CRNN 训练
│   ├── ocr_model.py                       #   模型定义 (MobileNetV3 + BiLSTM + CTC)
│   ├── test_training.py                   #   训练测试脚本
│   └── ppocr_keys_v1.txt                  #   字符字典 (0-9, B, C, J, T, U)
│
├── test_model/                            # 测试脚本
│   ├── test_wagon_number.py               #   车号模型测试 (YOLO + OCR + 端到端)
│   ├── test_debris.py                     #   异物/车门模型测试 (YOLO mAP)
│   ├── test_merged.py                     #   统一模型测试 (nc=6)
│   ├── camera_infer.py                    #   摄像头实时推理 (YOLO + OCR)
│   ├── convert_test_data_wagon_number.py  #   车号测试数据转换
│   └── results/                           #   测试结果
│       ├── wagon_number/                  #     车号测试报告
│       ├── carriage_rim_debris/           #     车厢沿异物测试报告
│       ├── track_intrusion/               #     轨道异物测试报告
│       ├── door_state/                    #     车门状态测试报告
│       └── merged/                        #     统一模型测试报告
│
└── docs/                                  # 项目文档
    ├── 项目目录结构说明.md
    ├── TBJU数据集转换规范.md
    ├── 异物数据扩充设计方案.md
    ├── 数据集转换流程与结果.md
    ├── OCR训练问题记录与解决方案.md
    └── ELF2_RK3588_...md                  #   RK3588 部署技术路线
```

---

## 3. 路径配置

`config.py` 使用 `Path(__file__).parent` 自动推导项目根目录，所有路径均为相对路径，换电脑无需修改。

```python
PROJECT_ROOT = Path(__file__).parent.resolve()   # Carriage_training_pipeline/
RAW_DATA_DIR  = PROJECT_ROOT / "raw_data"        # 原始数据
DATASETS_DIR  = PROJECT_ROOT / "datasets"        # 转换后的数据集
OUTPUT_DIR    = DATASETS_DIR / "output"          # train+val
TEST_OUTPUT   = DATASETS_DIR / "test_output"     # test
YOLO_OUTPUT   = PROJECT_ROOT / "YOLO_train" / "runs"  # YOLO 模型输出
OCR_OUTPUT    = PROJECT_ROOT / "OCR_train" / "output" # OCR 模型输出
```

---

## 4. 数据流

### 4.1 车号识别数据流

```
raw_data/eye_level/ + raw_data/side_view/
  │  (Label Studio JSON + 原始图片)
  ↓
scripts/convert_labelstudio_tbju.py
  │  解析 JSON → 文本标准化 → 坐标转换 → 裁剪 → 划分 train/val
  ↓
datasets/output/wagon_number_detection_平视/   (YOLO: images + txt labels)
datasets/output/wagon_number_detection_侧视/
datasets/output/wagon_number_ocr_平视/         (OCR: crops + labels.csv)
datasets/output/wagon_number_ocr_侧视/
  │
  ↓
YOLO_train/train_yolo_wagon_number.py → 合并平视+侧视 → 训练 → best.pt + best.onnx
OCR_train/train_ocr.py                           → 训练 → best_model.pth
  │
  ↓
test_model/test_wagon_number.py → mAP / OCR Accuracy / E2E Accuracy
  │
  ↓
ONNX → rknn-toolkit2 → RKNN → RK3588 部署
```

### 4.2 异物检测数据流（carriage_rim_debris, track_intrusion）

```
raw_data/{task}/                            Step 0: 原始图片
  │
  ↓ Label Studio 标注 region 区域
raw_data/{task}/train/labels/*.json         Step 1: 手动标注 JSON
  │
  ↓ scripts/augment_debris.py
raw_data/{task}/train/images/*_aug0.jpg     Step 2: 扩充图 (原图名 + _aug{N})
raw_data/{task}/train/labels/augmented_debris_train.json  (扩充标注)
  │
  ↓ scripts/setup_new_datasets.py
datasets/output/{task}_detection/           Step 3: 复制原图+扩充图
  │
  ↓ scripts/convert_labelstudio_new.py
datasets/output/{task}_detection/labels/*.txt  Step 4: YOLO txt 标签
  │
  ↓ YOLO_train/train_yolo_debris.py --task {task}
YOLO_train/runs/{task}_detection/weights/   Step 5: best.pt + best.onnx
  │
  ↓ test_model/test_debris.py --task {task}
test_model/results/{task}/test_report.txt   Step 6: 测试报告
```

**异物检测训练样本说明：**
- 原图 `0006_S_f0004.jpg` → 标签 `0006_S_f0004.txt`（只有 region）→ **负样本**（无异物）
- 扩充图 `0006_S_f0004_aug0.jpg` → 标签 `0006_S_f0004_aug0.txt`（region + debris）→ **正样本**（有异物）
- 两类样本都需要，比例约 1:1

### 4.3 车门状态数据流（door_state，无异物扩充）

```
raw_data/door_state/                        原始图片
  │
  ↓ Label Studio 标注 door_region
raw_data/door_state/train/labels/*.json     手动标注 JSON
  │
  ↓ scripts/setup_new_datasets.py           复制图片
  ↓ scripts/convert_labelstudio_new.py      JSON → YOLO txt (nc=1, class 0=region)
datasets/output/door_state_detection/
  │
  ↓ YOLO_train/train_yolo_debris.py --task door_state
YOLO_train/runs/door_state_detection/weights/best.{pt,onnx}
  │
  ↓ test_model/test_debris.py --task door_state
test_model/results/door_state/test_report.txt
```

车门状态只需 region 检测（nc=1），不经过 augment_debris.py。

---

## 5. 脚本说明

### 5.1 数据处理脚本 (scripts/)

| 脚本 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `convert_labelstudio_tbju.py` | raw_data/{eye_level,side_view} + Label Studio JSON | datasets/output/wagon_number_* | 车号 JSON → YOLO 检测 + OCR 裁剪 |
| `convert_labelstudio_new.py` | raw_data/{3个异物任务} + Label Studio JSON | datasets/output/{3个任务}_detection/labels/*.txt | 异物/车门 JSON → YOLO txt |
| `augment_debris.py` | raw_data/{task} + raw_data/debris_materials | raw_data/{task}/images/*_aug*.jpg + JSON | 异物贴图扩充（按分类基准尺寸，±10%浮动，Poisson分布控制数量） |
| `setup_new_datasets.py` | raw_data/{3个异物任务} | datasets/output/{3个任务}_detection/images/ | 清空旧图，复制原图+扩充图 |
| `merge_datasets.py` | datasets/output/{4个任务} | datasets/merged_detection/ | 合并四任务数据集（nc=6） |

**运行顺序：** augment_debris.py → setup_new_datasets.py → convert_labelstudio_new.py → merge_datasets.py

### 5.2 训练脚本 (YOLO_train/, OCR_train/)

| 脚本 | 任务 | 输出模型 |
|------|------|----------|
| `train_yolo_wagon_number.py` | 车号检测 (nc=1) | YOLO_train/runs/tbju_detection/weights/best.{pt,onnx} |
| `train_yolo_debris.py --task carriage_rim_debris` | 车厢沿异物 (nc=2) | YOLO_train/runs/carriage_rim_debris_detection/weights/best.{pt,onnx} |
| `train_yolo_debris.py --task track_intrusion` | 轨道异物侵限 (nc=2) | YOLO_train/runs/track_intrusion_detection/weights/best.{pt,onnx} |
| `train_yolo_debris.py --task door_state` | 车门状态 (nc=1) | YOLO_train/runs/door_state_detection/weights/best.{pt,onnx} |
| `train_yolo_merged.py` | 统一模型 (nc=6) | YOLO_train/runs/merged_detection/weights/best.{pt,onnx} |
| `train_ocr.py` | 车号 OCR 识别 | OCR_train/output/ppocr_rec_carriage_number/best_model.pth |

训练完成后自动导出 ONNX 模型（`--no_export` 可跳过）。

### 5.3 测试脚本 (test_model/)

| 脚本 | 任务 | 指标 |
|------|------|------|
| `test_wagon_number.py` | 车号 (YOLO+OCR+E2E) | mAP50, Precision, Recall, OCR Accuracy, E2E Accuracy |
| `test_debris.py --task {task}` | 异物/车门 (YOLO) | mAP50, mAP50-95, Precision, Recall, F1, 按类别 AP50/AP |
| `test_merged.py` | 统一模型 (nc=6) | mAP50, mAP50-95, Precision, Recall, F1, 按类别 AP50/AP |

---

## 6. Label Studio 标注格式

### 6.1 车号标注

```json
{
  "file_upload": "DJI_xxx.jpg",
  "annotations": [{
    "result": [
      {"id": "abc", "type": "rectanglelabels", "value": {"x": 81.8, "y": 35.5, "width": 11.8, "height": 4.3, "rectanglelabels": ["Carriage Number (Eye Level)"]}},
      {"id": "abc", "type": "textarea", "value": {"text": ["TBJU4950882"]}}
    ]
  }]
}
```

- 坐标为百分比 (0-100)
- 框和文本通过 `id` 字段关联
- 类别名：`Carriage Number (Eye Level)` 或 `Carriage Number (Side Level)`

### 6.2 异物/车门标注

```json
{
  "file_upload": "0006_S_f0004.jpg",
  "annotations": [{
    "result": [
      {"id": "xyz", "type": "rectanglelabels", "value": {"x": 10.5, "y": 60.0, "width": 80.0, "height": 30.0, "rectanglelabels": ["track_region"]}}
    ]
  }]
}
```

- 只标注 region（放置区域），debris 标注由 augment_debris.py 自动生成
- 类别名：`track_region` / `carriage_rim_region` / `door_region`

---

## 7. 数据集类别定义

| 数据集 | nc | class_id=0 | class_id=1 |
|--------|-----|------------|------------|
| 车号检测 | 1 | TBJU_region | — |
| 车厢沿异物 | 2 | region | debris |
| 轨道异物侵限 | 2 | region | debris |
| 车门状态 | 1 | region | — |
| 统一模型 | 6 | TBJU_region, region, debris (×4任务) | — |

YOLO 标签格式（每行一个框）：
```
class_id x_center y_center width height    # 归一化坐标 0-1
```

---

## 8. 完整工作流命令

```bash
# 0. 安装依赖
pip install -r requirements.txt
python config.py  # 验证路径

# 1. 车号数据转换
python scripts/convert_labelstudio_tbju.py --dataset_dirs raw_data/eye_level raw_data/side_view --output datasets/output

# 2. 异物扩充（基准尺寸已在代码 CATEGORY_BASE_SIZES 中配置，泊松分布控制数量）
python scripts/augment_debris.py --dataset_dir raw_data/carriage_rim_debris --debris_dir raw_data/debris_materials --class_name carriage_rim_region --debris_filter rock
python scripts/augment_debris.py --dataset_dir raw_data/track_intrusion --debris_dir raw_data/debris_materials --class_name track_region

# 3. 同步图片 + 转换标签
python scripts/setup_new_datasets.py
python scripts/convert_labelstudio_new.py

# 4. 训练
python YOLO_train/train_yolo_wagon_number.py --use_config
python YOLO_train/train_yolo_debris.py --task carriage_rim_debris --use_config
python YOLO_train/train_yolo_debris.py --task track_intrusion --use_config
python YOLO_train/train_yolo_debris.py --task door_state --use_config
python YOLO_train/train_yolo_merged.py --use_config
python OCR_train/train_ocr.py --use_config

# 5. 测试
python test_model/test_wagon_number.py --use_config --test_all
python test_model/test_debris.py --task carriage_rim_debris --use_config
python test_model/test_debris.py --task track_intrusion --use_config
python test_model/test_debris.py --task door_state --use_config
python test_model/test_merged.py --use_config

# 6. 部署（ONNX → RKNN）
# 将 best.onnx 拷贝到 TBJU_edge_inference_app/models/ 目录
# 使用 rknn-toolkit2 转换为 .rknn 格式
```

---

## 9. 输出路径

### 训练产物目录树

```
YOLO_train/runs/
├── README.md                                    # 本目录说明
├── tbju_detection/                              # 车号检测 (nc=1)
│   └── weights/{best.pt, best.onnx, last.pt}
├── carriage_rim_debris_detection/               # 车厢沿异物 (nc=2)
│   └── weights/{best.pt, best.onnx, last.pt}
├── track_intrusion_detection/                   # 轨道异物 (nc=2)
│   └── weights/{best.pt, best.onnx, last.pt}
├── door_state_detection/                        # 车门状态 (nc=1)
│   └── weights/{best.pt, best.onnx, last.pt}
└── merged_detection/                            # 统一模型 (nc=6)
    └── weights/{best.pt, best.onnx, last.pt}

OCR_train/output/
├── README.md
└── ppocr_rec_carriage_number/
    ├── best_model.pth                           # 最佳 PyTorch 模型
    ├── ppocr_rec_tbju.onnx                      # ONNX 模型（部署用）
    ├── training_log.json                        # 训练日志
    └── evaluation_results.csv                   # 评估结果
```

每个 `runs/{task}_detection/` 内含：weights/（模型）、results.csv（指标）、results.png（训练曲线）、confusion_matrix.png（混淆矩阵）、inference.py（推理脚本）。详见各目录下的 README.md。

### 测试产物目录树

```
test_model/results/
├── README.md                                    # 本目录说明
│
├── wagon_number/                                # 车号检测+识别
│   ├── test_report.txt                          #   汇总报告（YOLO+OCR+E2E）
│   ├── ocr_recognition_details.csv              #   OCR 详细结果
│   ├── ocr_errors.csv                           #   OCR 错误样本
│   ├── end_to_end_details.csv                   #   端到端详细结果
│   └── runs/detect/val-N/                       #   YOLO 验证可视化
│
├── carriage_rim_debris/                         # 车厢沿异物
│   ├── test_report.txt
│   ├── test_results.csv
│   └── runs/detect/val-N/
│
├── track_intrusion/                             # 轨道异物
│   ├── test_report.txt
│   ├── test_results.csv
│   └── runs/detect/val-N/
│
├── door_state/                                  # 车门状态
│   ├── test_report.txt
│   ├── test_results.csv
│   └── runs/detect/val-N/
│
└── merged/                                      # 统一模型
    ├── test_report.txt
    ├── test_results.csv
    └── runs/detect/val-N/
```

详见 `test_model/results/README.md`。

---

## 10. 测试指标

| 指标 | 说明 | 期望值 |
|------|------|--------|
| mAP50 | IoU=0.5 平均精度 | >0.9 |
| mAP50-95 | IoU=0.5:0.95 平均精度 | >0.7 |
| Precision | 正确检测 / 总检测 | >0.9 |
| Recall | 正确检测 / 总真实 | >0.9 |
| F1 | Precision 和 Recall 调和平均 | >0.9 |
| OCR Accuracy | 完全匹配率 | >0.95 |
| E2E Accuracy | 检测+识别都正确 | >0.9 |

---

## 11. 文档索引

| 文档 | 内容 |
|------|------|
| `docs/项目目录结构说明.md` | 目录结构与数据流详细说明 |
| `docs/TBJU数据集转换规范.md` | 车号数据转换规范（JSON 格式、坐标转换、文本校验） |
| `docs/异物数据扩充设计方案.md` | 异物扩充方案（区域标注、素材管理、贴图策略） |
| `docs/数据集转换流程与结果.md` | 转换过程记录与验证结果 |
| `docs/OCR训练问题记录与解决方案.md` | OCR 训练问题排查 |
| `docs/ELF2_RK3588_...md` | RK3588 板端部署技术路线（ONNX → RKNN） |

---

## 12. 依赖

```
ultralytics      # YOLOv8
torch            # PyTorch
Pillow           # 图像处理
opencv-python    # 视频处理
pyyaml           # YAML 配置
numpy            # 数值计算
```

---

## 13. 相关项目

- [TBJU_edge_inference_app](../TBJU_edge_inference_app/) — RK3588 边缘推理应用
- [TBJU-carriage-detection](../../TBJU-carriage-detection/) — GitHub 仓库主页
