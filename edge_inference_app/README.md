# TBJU_edge_inference_app — 列车车厢检测边缘推理应用

## 1. 项目概述

基于 YOLOv8 统一检测模型 (nc=6) + PP-OCR Rec 车号识别模型的**边缘推理应用**。
支持 **Windows 11 PC**（开发/测试）和 **ELF2 / RK3588 开发板**（部署/展示）双平台运行。

训练管道见 [Carriage_training_pipeline](../Carriage_training_pipeline/)。

**核心功能：**
- YOLOv8 统一检测（6 类：车号区域 + 4 个任务区域 + 异物）
- PP-OCR Rec 车号识别（15 字符：0-9, B, C, J, T, U）
- PyQt5 GUI 界面（三标签页：检测/监控/压测）
- LD3320 UART 语音控制
- 声音报警系统
- 事件上传 + 远程看板
- 系统性能监控

**技术栈：** Python, YOLOv8, ONNX, rknn-toolkit-lite2, PyQt5, FastAPI

---

## 2. 目录结构

```
TBJU_edge_inference_app/
├── run_gui.py                    # GUI 启动入口（PyQt5，自动回退 tkinter）
├── run_smoke_test.py             # 冒烟测试入口
├── watch_csv.py                  # CSV 监控 + TXT 摘要自动生成
├── start_gui_with_audio.sh       # 板端启动脚本（含音频设备检测）
├── requirements.txt              # Python 依赖
├── README.md                     # 本文件
│
├── src/                          # 源码
│   ├── core/
│   │   └── tbju_rknn_core.py     # 推理核心（RKNN/PyTorch/ONNX 三后端，带推理锁）
│   ├── gui/
│   │   ├── tbju_demo_gui.py      # PyQt5 GUI 主程序（三标签页 + RecordingController）
│   │   └── tbju_demo_tk.py       # tkinter 回退 GUI
│   ├── monitor/
│   │   └── tbju_system_monitor.py # 系统监控（线程安全，CPU/GPU/NPU 三源采集）
│   ├── voice/
│   │   └── ld3320_uart.py        # LD3320 UART 语音控制
│   ├── alarm/
│   │   └── voice_alarm.py        # 声音报警模块（queue maxsize=50）
│   ├── capacity/
│   │   └── tbju_capacity_test.py  # 扩展能力压测（1/2/4 ROI，流式 CSV）
│   └── network/
│       ├── utils.py              # 公共网络工具（get_local_ipv4, validate_url）
│       ├── event_uploader.py     # 事件上传（显式 start，200MB 磁盘限额，CSV 同步）
│       └── command_poller.py     # 远程命令轮询（KeyError 保护，ack 基于返回值）
│
├── scripts/                      # CLI 脚本
│   ├── inference_tbju.py          # 单图推理
│   ├── inference_tbju_stream.py   # 视频/摄像头推理
│   ├── inference_tbju_merged.py   # 统一模型推理（备份）
│   ├── voice_alarm_test.py        # 声音报警 CLI 测试
│   └── voice_ld3320_test.py       # LD3320 串口测试
│
├── models/                       # 模型文件（需从训练管道导出）
│   ├── yolo/
│   │   ├── merged_yolov8.rknn    # YOLO 默认 (FP16, 23.4 MB)
│   │   ├── merged_yolov8_fp.rknn # YOLO FP16 备份
│   │   └── merged_yolov8_i8.rknn # YOLO INT8（速度优先, 12.4 MB）
│   └── ocr/
│       ├── rec_tbju.rknn         # OCR 默认 (4.2 MB)
│       └── rec_tbju_fp.rknn      # OCR FP16 备份
│
├── config/                       # 配置文件
│   ├── merged_classes.txt         # 6 类定义
│   ├── ppocr_keys_v1.txt         # OCR 字符字典 (15 字符)
│   └── voice_commands.json       # 语音命令映射
│
├── tests/                        # 测试
│   ├── smoke_core.py             # 推理核心冒烟测试
│   └── test_core.py              # pytest 单元测试（27 项）
│
├── tools/                        # 工具
│   ├── rknn_toolkit_lite2-*.whl  # RKNN Lite2 安装包
│   └── pyqt5_wheels/             # PyQt5 离线 wheel
│
├── assets/                       # 资源文件
│   └── audio/                    # 声音报警音频
│
├── docs/                         # 文档
│   ├── BOARD_RUN_CHECKLIST.md    # 板端部署清单
│   ├── LD3320_VOICE_CONTROL.md   # 语音控制文档
│   ├── LD3320_STC11_WINDOWS_DEBUG.md  # Windows 调试文档
│   ├── README_GUI_DEMO.md        # GUI 使用说明
│   └── README_MERGED_RKNN.txt    # RKNN 部署说明
│
├── tbju-dashboard/               # 远程看板（FastAPI Web 应用，3 Tab 布局）
│   ├── app.py                    # FastAPI 主应用（带路径穿越防护 + 带宽中间件）
│   ├── database.py               # SQLite 数据库（WAL + busy_timeout=5000）
│   ├── requirements.txt          # 依赖
│   └── static/                   # 前端页面（深色工业风）
│       ├── index.html
│       ├── script.js
│       └── style.css
│
└── output/                       # 输出目录（自动创建）
    ├── images/                   # 图片推理结果
    ├── videos/                   # 视频推理结果
    ├── camera/                   # 摄像头推理结果
    ├── capacity/                 # 压测报告
    ├── logs/                     # 系统监控日志
    └── network/events/           # 事件上传队列
```

---

## 3. 快速启动

### Windows PC

```bash
# 安装依赖
pip install -r requirements.txt
# 取消注释 requirements.txt 中的 Windows 专用依赖
pip install ultralytics torch onnxruntime

# 启动 GUI
python run_gui.py

# 冒烟测试
python run_smoke_test.py
```

### RK3588 开发板

```bash
cd /home/elf/TBJU_edge_inference_app

# 安装依赖
pip install -r requirements.txt
pip install tools/rknn_toolkit_lite2-*.whl
pip install pyserial

# 启动 GUI
export DISPLAY=:0.0
python3 run_gui.py

# 或使用含音频检测的启动脚本
bash start_gui_with_audio.sh
```

### 远程看板

```bash
cd tbju-dashboard
pip install -r requirements.txt
python app.py
# 访问 http://localhost:8000
```

### CSV 监控

```bash
python watch_csv.py
# 自动监控 output/ 目录，为新 CSV 生成 TXT 摘要
```

---

## 4. GUI 三标签页

| Tab | 名称 | 功能 |
|-----|------|------|
| Tab 1 | 检测识别 | 图片/视频/摄像头检测、语音控制、声音报警、事件上传 |
| Tab 2 | 性能监控 | 实时系统资源监控（CPU/内存/温度/NPU/GPU），历史曲线与表格 |
| Tab 3 | 扩展能力 | 1/2/4 ROI 压力测试，输出容量报告 |

---

## 5. 远程看板 3 Tab 布局

| Tab | 名称 | 内容 |
|-----|------|------|
| Tab1 | 总览 | KPI 卡片 + 告警 + 远程控制 + 日志/命令历史 + 最近事件 + 设备状态 |
| Tab2 | 检测 | 同步文件列表 + 事件表格（7 列）+ 操作栏 |
| Tab3 | 性能 | 6 数值卡片（CPU/内存/温度/推理耗时/NPU/GPU）+ 7 图表 |

---

## 6. CSV 同步流程

```
板端检测结束 → _sync_csv_to_dashboard（扫描 session + logs 目录）
  → POST /api/files/upload（带 source_dir）
    → 看板保存到 output/{source_dir}/{session名}/（与开发板目录对齐）
      → watch_csv.py 自动生成同目录下的 TXT 摘要

PC 端 output/ 目录结构（与开发板一致）：
  output/camera/日期_时间/result.csv + result.txt
  output/images/日期_时间/result.csv + result.txt
  output/videos/日期_时间/result.csv + result.txt
  output/logs/metrics_*.csv + metrics.txt
```

---

## 7. 安全特性

- `torch.load(weights_only=True)` 防 pickle 反序列化
- `TBJURKNNEngine._infer_lock` 推理锁防并发
- `EventUploader` 显式 `start()` + 200MB 磁盘限额
- `safe_csv_cell()` 防 CSV 注入
- 看板 `/api/files/download` 路径穿越防护
- SQLite `BEGIN IMMEDIATE` + `busy_timeout=5000`
- 语音报警队列 `maxsize=50`
- 命令 ack 基于回调真实返回值

---

## 8. 测试

```bash
# pytest 单元测试（27 项）
python tests/test_core.py

# 冒烟测试（4 项）
python run_smoke_test.py
```

---

## 9. 依赖

### 主应用

```bash
# 通用依赖
pip install opencv-python numpy PyQt5 psutil

# Windows 专用
pip install ultralytics torch onnxruntime

# RK3588 专用
pip install tools/rknn_toolkit_lite2-*.whl
pip install pyserial
```

### 远程看板

```bash
cd tbju-dashboard
pip install fastapi uvicorn python-multipart
```

---

## 10. 文档索引

| 文档 | 内容 |
|------|------|
| `docs/README_GUI_DEMO.md` | GUI 使用说明（图片/视频/摄像头/语音/报警/上传） |
| `docs/BOARD_RUN_CHECKLIST.md` | ELF2/RK3588 板端部署清单 |
| `docs/LD3320_VOICE_CONTROL.md` | LD3320 语音控制详细文档 |
| `docs/LD3320_STC11_WINDOWS_DEBUG.md` | Windows 串口调试文档 |

---

## 11. 相关项目

- [Carriage_training_pipeline](../Carriage_training_pipeline/) — 模型训练管道
- [TBJU-carriage-detection](../../TBJU-carriage-detection/) — GitHub 仓库主页
