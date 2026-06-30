# TBJU-carriage-detection

列车车厢视觉检测与识别系统，基于深度学习，部署目标为 RK3588（ELF2）开发板。

## 🎯 项目亮点

- **统一检测模型**：YOLOv8 (nc=6) 一次推理完成 6 类检测
- **双平台支持**：Windows PC 开发 + RK3588 嵌入式部署
- **完整工作流**：数据标注 → 训练 → 测试 → 部署全链路
- **智能交互**：GUI 界面 + 语音控制 + 声音报警 + 远程看板

---

## 📋 检测任务

| 任务 | 模型 | 类别 | 说明 |
|------|------|------|------|
| 车号检测 | YOLOv8 (nc=1) | TBJU_region | 定位车号区域 |
| 车号识别 | PP-OCR Rec (CRNN+CTC) | — | 识别车号文本（TBJU + 7位数字） |
| 车厢沿异物 | YOLOv8 (nc=2) | region, debris | 检测车厢沿上的异物 |
| 轨道异物侵限 | YOLOv8 (nc=2) | region, debris | 检测轨道区域内的异物 |
| 车门状态 | YOLOv8 (nc=1) | region | 检测车门区域 |
| 统一模型 | YOLOv8 (nc=6) | 6类 | 合并四任务，一次推理完成所有检测 |

**技术栈：** Python, YOLOv8, PyTorch, Label Studio, ONNX, rknn-toolkit2, PyQt5, FastAPI

---

## 📁 目录结构

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
    ├── run_gui.py                     # GUI 启动入口
    ├── run_smoke_test.py              # 冒烟测试
    ├── watch_csv.py                   # CSV 监控
    ├── start_gui_with_audio.sh        # 板端启动脚本
    ├── requirements.txt               # Python 依赖
    ├── src/                           # 核心模块 (7个子模块)
    │   ├── core/                      # 推理核心（RKNN/PyTorch/ONNX 三后端）
    │   ├── gui/                       # GUI 界面（PyQt5/tkinter）
    │   ├── alarm/                     # 声音报警
    │   ├── network/                   # 事件上传 + 命令轮询
    │   ├── voice/                     # LD3320 语音控制
    │   ├── monitor/                   # 系统监控（CPU/GPU/NPU）
    │   └── capacity/                  # 压力测试
    ├── scripts/                       # CLI 脚本 (5个)
    ├── tests/                         # 单元测试 (67项)
    ├── config/                        # 配置文件
    ├── models/                        # 预训练模型（RKNN 格式）
    │   ├── yolo/                      # YOLO 检测模型
    │   └── ocr/                       # OCR 识别模型
    ├── tools/                         # 离线安装包
    │   ├── rknn_toolkit_lite2-*.whl   # RKNN Lite2 安装包
    │   └── pyqt5_wheels/              # PyQt5 离线包
    ├── tbju-dashboard/                # Web 看板（FastAPI）
    ├── assets/                        # 资源文件
    └── docs/                          # 部署文档 (5个)
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/wanmengtao/TBJU-carriage-detection.git
cd TBJU-carriage-detection
```

### 2. RK3588 开发板 — 运行推理应用

推理应用（GUI）运行在 RK3588 开发板上，用于实时检测：

```bash
cd TBJU_edge_inference_app

# 安装依赖
pip install -r requirements.txt
pip install tools/rknn_toolkit_lite2-*.whl
pip install pyserial

# 启动 GUI
export DISPLAY=:0.0
python3 run_gui.py

# 或使用含音频检测的启动脚本
bash start_gui_with_audio.sh

# 运行冒烟测试
python run_smoke_test.py
```

### 3. Windows PC — 启动远程看板

远程看板运行在 Windows PC 上，用于接收开发板上传的检测事件和数据：

```bash
cd TBJU_edge_inference_app/tbju-dashboard

# 安装依赖
pip install -r requirements.txt

# 启动看板服务
python app.py
# 访问 http://localhost:8000
```

**看板功能：**
- Tab1：总览（KPI + 告警 + 远程控制 + 日志）
- Tab2：检测（事件表格 + 操作）
- Tab3：性能（CPU/内存/温度/推理耗时监控）

**使用方式：**
1. 在 PC 上启动看板服务
2. 在开发板 GUI 中填写看板地址（如 `http://PC端IP:8000/api/events`）
3. 勾选"启用上传"，点击"测试连接"
4. 开发板检测到事件时自动上传到看板

### 4. 训练模型（可选）

如需重新训练模型，在 Windows PC 上运行训练管道：

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

---

## 📦 模型文件

本仓库包含预训练的 RKNN 模型文件，可直接在 RK3588 上运行：

| 模型 | 文件 | 大小 | 说明 |
|------|------|------|------|
| YOLO 默认 | `models/yolo/merged_yolov8.rknn` | ~23MB | FP16 精度，推荐使用 |
| YOLO FP16 | `models/yolo/merged_yolov8_fp.rknn` | ~23MB | FP16 备份 |
| YOLO INT8 | `models/yolo/merged_yolov8_i8.rknn` | ~12MB | INT8 量化，速度优先 |
| OCR 默认 | `models/ocr/rec_tbju.rknn` | ~4MB | FP16 精度 |
| OCR FP16 | `models/ocr/rec_tbju_fp.rknn` | ~4MB | FP16 备份 |

**模型说明：**
- YOLO 模型：统一检测 6 类（车号区域 + 4个任务区域 + 异物）
- OCR 模型：识别车号文本（15 字符：0-9, B, C, J, T, U）
- 格式：RKNN（RK3588 NPU 专用格式）

---

## 🎮 功能特性

### GUI 三标签页

| Tab | 功能 | 说明 |
|-----|------|------|
| Tab 1 | 检测识别 | 图片/视频/摄像头检测、语音控制、声音报警、事件上传 |
| Tab 2 | 性能监控 | CPU/内存/温度/NPU/GPU 实时监控，历史曲线 |
| Tab 3 | 扩展能力 | 1/2/4 ROI 压力测试，输出容量报告 |

### 语音控制（LD3320）

支持语音命令控制：
- 打开摄像头、开始/停止/暂停检测
- 开始/停止评估、开始/停止录制
- 系统状态、静音/解除静音

### 声音报警

检测到异物或系统异常时，自动通过 USB 小音箱语音播报。

### 远程看板

FastAPI Web 应用，3 Tab 布局：
- Tab1：总览（KPI + 告警 + 控制 + 日志）
- Tab2：检测（事件表格 + 操作）
- Tab3：性能（6 数值卡片 + 7 图表）

### 核心算法

**区域约束 + 短时多帧确认**：

```
YOLO 推理 → decode_yolov8()
         → validate_debris_region()     ← 区域验证（不改类别）
         → OCR（车号区域）
         → TemporalConsistencyFilter    ← 滑动窗口内命中 M 次确认
         → draw_detections()
```

- **区域验证**：异物落在对应区域内才保留，过滤背景误报
- **时序一致性**：滑动窗口 5 帧，至少出现 3 次才确认告警
- **类名校验**：支持乱序显式 ID，校验重复、缺号、格式混用

---

## 📊 测试指标

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

## 🔧 工作流程

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

## 📚 文档

| 文档 | 内容 |
|------|------|
| [训练管道说明](Carriage_training_pipeline/README.md) | 数据处理、训练、测试详细说明 |
| [GUI 使用说明](TBJU_edge_inference_app/docs/README_GUI_DEMO.md) | 图片/视频/摄像头检测、语音控制 |
| [板端部署清单](TBJU_edge_inference_app/docs/BOARD_RUN_CHECKLIST.md) | ELF2/RK3588 部署步骤 |
| [语音控制文档](TBJU_edge_inference_app/docs/LD3320_VOICE_CONTROL.md) | LD3320 接线与配置 |
| [数据集转换规范](Carriage_training_pipeline/docs/TBJU数据集转换规范.md) | Label Studio JSON 格式 |
| [异物扩充方案](Carriage_training_pipeline/docs/异物数据扩充设计方案.md) | 异物贴图扩充策略 |

---

## 💻 依赖

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

## 🎬 演示视频

[待添加]

---

## 📥 数据集与模型下载

数据集和训练好的 YOLO 模型文件较大，通过百度网盘分享：

| 内容 | 大小 | 说明 |
|------|------|------|
| datasets | ~2.2GB | 转换后的数据集（YOLO 训练用） |
| YOLO 模型 | ~3.5GB | 训练好的 YOLO 检测模型（所有任务） |

**百度网盘下载：**
- 链接：https://pan.baidu.com/s/1U6slzn5Ij__8QQvn9hsJyA
- 提取码：`nqhr`
- 文件：`datasets_yolomodel.rar`

**OCR 模型：** 包含在本仓库的 `Carriage_training_pipeline/OCR_train/output/` 目录中（~20MB）。

**RKNN 模型：** 包含在本仓库的 `TBJU_edge_inference_app/models/` 目录中（~68MB）。

---

## 📝 开发日志

- 2026-06-30：初始版本发布
- 完成 5 个子任务 + 1 个统一模型
- 支持 Windows PC + RK3588 双平台
- 集成 GUI、语音控制、声音报警、远程看板

---

## 🤝 致谢

- YOLOv8：[ultralytics](https://github.com/ultralytics/ultralytics)
- PP-OCR：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Label Studio：[HumanSignal](https://github.com/HumanSignal/label-studio)

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

Copyright (c) 2026 wanmengtao

---

## 📧 联系方式

- 邮箱：1535121687@qq.com
