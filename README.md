# TBJU-carriage-detection

列车车厢视觉检测与识别系统，基于深度学习，部署目标为 RK3588（ELF2）开发板。

## 项目概述

本项目包含两个子模块：

| 子目录 | 功能 | 运行环境 |
|--------|------|----------|
| `training_pipeline/` | 模型训练管道 | PC (Windows/Linux) |
| `edge_inference_app/` | 边缘推理应用 | RK3588 / PC |

**检测任务：**

| 任务 | 模型 | 类别 | 说明 |
|------|------|------|------|
| 车号检测 | YOLOv8 (nc=1) | TBJU_region | 定位车号区域 |
| 车号识别 | PP-OCR Rec (CRNN+CTC) | — | 识别车号文本 |
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
├── training_pipeline/                 # 训练管道
│   ├── config.py                      # 路径配置
│   ├── requirements.txt               # Python 依赖
│   ├── README.md                      # 训练管道说明
│   ├── scripts/                       # 数据处理脚本
│   ├── YOLO_train/                    # YOLO 训练
│   ├── OCR_train/                     # OCR 训练
│   ├── test_model/                    # 测试脚本
│   └── docs/                          # 技术文档
│
└── edge_inference_app/                # 边缘推理应用
    ├── run_gui.py                     # GUI 启动
    ├── run_smoke_test.py              # 冒烟测试
    ├── src/                           # 核心模块
    │   ├── core/                      # 推理核心
    │   ├── gui/                       # GUI 界面
    │   ├── alarm/                     # 声音报警
    │   ├── network/                   # 网络上传
    │   ├── voice/                     # 语音控制
    │   ├── monitor/                   # 系统监控
    │   └── capacity/                  # 压力测试
    ├── scripts/                       # CLI 脚本
    ├── tests/                         # 单元测试
    ├── tbju-dashboard/                # Web 看板
    └── docs/                          # 部署文档
```

---

## 快速开始

### 训练管道

```bash
cd training_pipeline

# 安装依赖
pip install -r requirements.txt

# 验证路径
python config.py

# 车号数据转换
python scripts/convert_labelstudio_tbju.py --dataset_dirs raw_data/eye_level raw_data/side_view --output datasets/output

# 训练
python YOLO_train/train_yolo_wagon_number.py --use_config
python YOLO_train/train_yolo_debris.py --task carriage_rim_debris --use_config
python OCR_train/train_ocr.py --use_config
```

### 边缘推理应用

```bash
cd edge_inference_app

# Windows
python run_gui.py

# RK3588
export DISPLAY=:0.0
python3 run_gui.py

# 冒烟测试
python run_smoke_test.py
```

---

## 文档

- [训练管道说明](training_pipeline/README.md)
- [GUI 使用说明](edge_inference_app/docs/README_GUI_DEMO.md)
- [板端部署清单](edge_inference_app/docs/BOARD_RUN_CHECKLIST.md)
- [语音控制文档](edge_inference_app/docs/LD3320_VOICE_CONTROL.md)

---

## 许可证

[待定]
