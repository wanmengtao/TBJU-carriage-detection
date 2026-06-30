# 测试结果目录

按任务分类存放所有模型的测试报告和详细结果。

---

## 目录树

```
results/
├── README.md                                      # 本文件
│
├── wagon_number/                                  # 车号检测+识别测试
│   ├── test_report.txt                            #   汇总报告（YOLO+OCR+E2E）
│   ├── ocr_recognition_details.csv                #   OCR 每张 crop 识别结果
│   ├── ocr_errors.csv                             #   OCR 识别错误样本
│   ├── end_to_end_details.csv                     #   端到端测试详细结果
│   └── runs/detect/val-N/                         #   YOLO 验证可视化
│       ├── BoxF1_curve.png                        #     F1 曲线
│       ├── BoxPR_curve.png                        #     PR 曲线
│       ├── confusion_matrix.png                   #     混淆矩阵
│       └── val_batch*_labels/pred.jpg             #     验证 batch 对比
│
├── carriage_rim_debris/                           # 车厢沿异物测试
│   ├── test_report_carriage_rim_debris.txt        #   测试报告
│   └── test_results_carriage_rim_debris.csv       #   指标 CSV
│
├── track_intrusion/                               # 轨道异物侵限测试
│   ├── test_report_track_intrusion.txt
│   └── test_results_track_intrusion.csv
│
├── door_state/                                    # 车门状态测试
│   ├── test_report_door_state.txt
│   └── test_results_door_state.csv
│
└── merged/                                        # 统一模型测试
    ├── test_report_merged.txt
    └── test_results_merged.csv
```

## 各任务说明

| 目录 | 任务 | 测试脚本 | 状态 |
|------|------|----------|------|
| `wagon_number/` | 车号检测+OCR+端到端 | `test_wagon_number.py` | ✅ 已完成 |
| `carriage_rim_debris/` | 车厢沿异物检测 | `test_debris.py --task carriage_rim_debris` | ✅ 已完成 |
| `track_intrusion/` | 轨道异物侵限检测 | `test_debris.py --task track_intrusion` | ✅ 已完成 |
| `door_state/` | 车门状态检测 | `test_debris.py --task door_state` | ✅ 已完成 |
| `merged/` | 统一模型（nc=6） | `test_merged.py` | ✅ 已完成 |

## 车号测试结果说明

| 文件 | 内容 |
|------|------|
| `test_report.txt` | YOLO mAP + OCR 准确率 + 端到端准确率汇总 |
| `ocr_recognition_details.csv` | 每张 crop：预测文本、真实文本、是否匹配 |
| `ocr_errors.csv` | OCR 识别错误的 crop 列表 |
| `end_to_end_details.csv` | YOLO 检测框 + OCR 识别文本 vs 真实标注 |
| `runs/detect/val-N/` | ultralytics 验证输出（PR 曲线、混淆矩阵、batch 可视化） |

## 异物/车门测试结果说明（待生成）

| 文件 | 内容 |
|------|------|
| `test_report.txt` | mAP50、mAP50-95、Precision、Recall、F1、按类别 AP50/AP |
| `test_results.csv` | 同上指标的 CSV 格式 |
| `runs/detect/val-N/` | ultralytics 验证输出 |

## 测试命令

```bash
# 车号模型
python test_model/test_wagon_number.py --use_config --test_all

# 异物/车门模型
python test_model/test_debris.py --task carriage_rim_debris --use_config
python test_model/test_debris.py --task track_intrusion --use_config
python test_model/test_debris.py --task door_state --use_config

# 统一模型
python test_model/test_merged.py --use_config

# 摄像头实时推理
python test_model/camera_infer.py --model YOLO_train/run_merged/train/weights/best.pt --ocr
```
