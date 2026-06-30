# TBJU-carriage-detection

列车车厢视觉检测与识别系统，基于深度学习，部署目标为 RK3588（ELF2）开发板。

## 项目概述

本项目包含两个子模块：

| 子目录 | 功能 | 运行环境 |
|--------|------|----------|
| `Carriage_training_pipeline/` | 模型训练管道（数据处理、训练、测试） | PC (Windows/Linux) |
| `TBJU_edge_inference_app/` | 边缘推理应用（GUI、语音、报警、看板） | RK3588 / PC |

**检测任务：**

| 任务 | 模型 | 类别 | 说明 |
|------|------|------|------|
| 车号检测 | YOLOv8 (nc=1) | TBJU_region | 定位车号区域 |
| 车号识别 | PP-OCR Rec (CRNN+CTC) | — | 识别车号文本（TBJU + 7位数字） |
| 车厢沿异物 | YOLOv8 (nc=2) | region, debris | 检测车厢沿上的异物 |
| 轨道异物侵限 | YOLOv8 (nc=2) | region, debris | 检测轨道区域内的异物 |
| 车门状态 | YOLOv8 (nc=1) | region | 检测车门区域 |
| 统一模型 | YOLOv8 (nc=6) | 6类 | 合并四任务 |

**技术栈：** Python, YOLOv8, PyTorch, Label Studio, ONNX, rknn-toolkit2

---

## 目录结构

```
TBJU-carriage-detection/
├── README.md                          # 本文件
├── .gitignore                         # Git 忽略规则
│
├── Carriage_training_pipeline/        # 训练管道
│   ├── config.py                      # 路径配置
│   ├── requirements.txt               # Python 依赖
│   ├── README.md                      # 训练管道说明
│   ├── scripts/                       # 数据处理脚本 (6个)
│   ├── YOLO_train/                    # YOLO 训练 (3个)
│   ├── OCR_train/                     # OCR 训练 (4个)
│   ├── test_model/                    # 测试脚本 + 结果
│   └── docs/                          # 技术文档 (6个)
│
└── TBJU_edge_inference_app/           # 边缘推理应用
    ├── run_gui.py                     # GUI 启动
    ├── run_smoke_test.py              # 冒烟测试
    ├── watch_csv.py                   # CSV 监控
    ├── start_gui_with_audio.sh        # 板端启动脚本
    ├── requirements.txt               # Python 依赖
    ├── src/                           # 核心模块 (7个子模块)
    │   ├── core/                      # 推理核心
    │   ├── gui/                       # GUI 界面
    │   ├── alarm/                     # 声音报警
    │   ├── network/                   # 网络上传
    │   ├── voice/                     # 语音控制
    │   ├── monitor/                   # 系统监控
    │   └── capacity/                  # 压力测试
    ├── scripts/                       # CLI 脚本 (5个)
    ├── tests/                         # 单元测试
    ├── config/                        # 配置文件
    ├── tbju-dashboard/                # Web 看板
    ├── assets/                        # 资源文件
    └── docs/                          # 部署文档 (5个)
```

---

## 快速开始

### 训练管道

```bash
cd Carriage_training_pipeline

# 安装依赖
pip install -r requirements.txt

# 验证路径
python config.py

# 车号数据转换
python scripts/convert_labelstudio_tbju.py --dataset_dirs raw_data/eye_level raw_data/side_view --output datasets/output

# 训练
python YOLO_train/train_yolo_wagon_number.py --use_config
python YOLO_train/train_yolo_debris.py --task carriage_rim_debris --use_config
python YOLO_train/train_yolo_merged.py --use_config
python OCR_train/train_ocr.py --use_config

# 测试
python test_model/test_wagon_number.py --use_config --test_all
python test_model/test_merged.py --use_config
```

### 边缘推理应用

```bash
cd TBJU_edge_inference_app

# Windows
pip install -r requirements.txt
python run_gui.py

# RK3588
pip install -r requirements.txt
pip install tools/rknn_toolkit_lite2-*.whl
export DISPLAY=:0.0
python3 run_gui.py

# 冒烟测试
python run_smoke_test.py

# 远程看板
cd tbju-dashboard
pip install -r requirements.txt
python app.py
```

---

## 工作流程

```
1. 数据准备
   raw_data/ → Label Studio 标注 → scripts/ 转换 → datasets/

2. 模型训练
   datasets/ → YOLO_train/ + OCR_train/ → models/

3. 模型测试
   models/ → test_model/ → test results

4. 模型导出
   .pt → .onnx → .rknn (rknn-toolkit2)

5. 边缘部署
   .rknn → TBJU_edge_inference_app/models/ → run_gui.py
```

---

## 文档

| 文档 | 内容 |
|------|------|
| [训练管道说明](Carriage_training_pipeline/README.md) | 数据处理、训练、测试详细说明 |
| [GUI 使用说明](TBJU_edge_inference_app/docs/README_GUI_DEMO.md) | 图片/视频/摄像头检测、语音控制 |
| [板端部署清单](TBJU_edge_inference_app/docs/BOARD_RUN_CHECKLIST.md) | ELF2/RK3588 部署步骤 |
| [语音控制文档](TBJU_edge_inference_app/docs/LD3320_VOICE_CONTROL.md) | LD3320 接线与配置 |
| [数据集转换规范](Carriage_training_pipeline/docs/TBJU数据集转换规范.md) | Label Studio JSON 格式 |
| [异物扩充方案](Carriage_training_pipeline/docs/异物数据扩充设计方案.md) | 异物贴图扩充策略 |

---

## 测试指标

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

## 依赖

### 训练管道

```
ultralytics      # YOLOv8
torch            # PyTorch
Pillow           # 图像处理
opencv-python    # 视频处理
pyyaml           # YAML 配置
numpy            # 数值计算
```

### 边缘推理应用

```
# 通用
opencv-python, numpy, PyQt5, psutil

# Windows
ultralytics, torch, onnxruntime

# RK3588
rknn-toolkit-lite2, pyserial

# 看板
fastapi, uvicorn, python-multipart
```

---

## 许可证

[待定]
