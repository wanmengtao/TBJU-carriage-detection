# 模型测试

## 功能说明

测试训练好的 YOLO 检测模型和 OCR 识别模型，生成详细测试报告。

## 文件结构

```
test_model/
├── convert_test_data_wagon_number.py   # 车号测试数据转换（JSON → YOLO+OCR）
├── test_wagon_number.py                # 车号模型测试（YOLO+OCR+端到端）
├── test_debris.py                      # 异物/车门模型测试（YOLO 检测）
├── test_merged.py                      # 统一模型测试（nc=6）
├── camera_infer.py                     # 摄像头实时推理（YOLO+OCR）
├── results/                            # 测试结果输出目录
│   ├── wagon_number/                   #   车号检测+识别测试
│   ├── carriage_rim_debris/            #   车厢沿异物测试
│   ├── track_intrusion/                #   轨道异物侵限测试
│   ├── door_state/                     #   车门状态测试
│   └── merged/                         #   统一模型测试
└── README.md
```

---

## 车号检测测试

### 第一步：转换测试数据

```bash
python test_model/convert_test_data_wagon_number.py --use_config
```

### 第二步：运行测试

```bash
# 测试所有（YOLO + OCR + 端到端）
python test_model/test_wagon_number.py --use_config --test_all

# 只测试 YOLO
python test_model/test_wagon_number.py --use_config --test_yolo

# 只测试 OCR
python test_model/test_wagon_number.py --use_config --test_ocr
```

---

## 异物/车门检测测试

三个任务共用同一个脚本，通过 `--task` 参数区分：

```bash
# 车厢沿异物
python test_model/test_debris.py --task carriage_rim_debris --use_config

# 轨道异物侵限
python test_model/test_debris.py --task track_intrusion --use_config

# 车门状态
python test_model/test_debris.py --task door_state --use_config
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --task | 任务名 (carriage_rim_debris / track_intrusion / door_state) | 必填 |
| --use_config | 使用 config.py 中的路径配置 | False |
| --model_path | 模型路径（不用 config 时可指定） | — |
| --dataset_yaml | 数据集 YAML 路径（不用 config 时可指定） | — |
| --img_size | 输入图片尺寸 | 640 |
| --batch_size | 批量大小 | 16 |
| --device | 计算设备 | 0 |
| --iou | IoU 阈值 | 0.5 |
| --output | 输出目录 | test_model/results |

---

## 统一模型测试

测试合并后的 6 类别 YOLO 模型：

```bash
python test_model/test_merged.py --use_config
python test_model/test_merged.py --use_config --split val
```

---

## 摄像头实时推理

使用训练好的模型进行实时检测：

```bash
# 只用 YOLO 检测
python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt

# YOLO + OCR 车号识别
python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt --ocr

# 使用视频文件
python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt --source video.mp4

# 保存结果视频
python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt --save result.mp4
```

---

## 输出说明

测试结果按任务分类保存在 `test_model/results/`：

```
results/
├── wagon_number/                      # 车号
│   ├── test_report.txt
│   ├── ocr_recognition_details.csv
│   ├── ocr_errors.csv
│   └── end_to_end_details.csv
├── carriage_rim_debris/               # 车厢沿异物
│   ├── test_report_carriage_rim_debris.txt
│   └── test_results_carriage_rim_debris.csv
├── track_intrusion/                   # 轨道异物
│   ├── test_report_track_intrusion.txt
│   └── test_results_track_intrusion.csv
├── door_state/                        # 车门状态
│   ├── test_report_door_state.txt
│   └── test_results_door_state.csv
└── merged/                            # 统一模型
    ├── test_report_merged.txt
    └── test_results_merged.csv
```

---

## 测试指标说明

### YOLO 检测指标

| 指标 | 说明 |
|------|------|
| mAP50 | IoU=0.5 时的平均精度 |
| mAP50-95 | IoU=0.5:0.95 时的平均精度 |
| Precision | 检测正确的框 / 总检测框数 |
| Recall | 检测正确的框 / 总真实框数 |
| F1 Score | Precision 和 Recall 的调和平均 |

### OCR 识别指标

| 指标 | 说明 |
|------|------|
| Accuracy | 完全匹配的样本数 / 总样本数 |
| 按视角统计 | 平视/侧视分别统计 |

### 端到端指标

| 指标 | 说明 |
|------|------|
| E2E Accuracy | YOLO 检测 + OCR 识别都正确的比例 |
