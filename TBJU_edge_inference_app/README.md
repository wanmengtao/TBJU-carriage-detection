# TBJU_edge_inference_app — 列车车厢检测边缘推理应用

## 1. 项目概述

基于 YOLOv8 统一检测模型 (nc=6) + PP-OCR Rec 车号识别模型的**边缘推理应用**。

**平台分工：**
- **ELF2 / RK3588 开发板**：运行推理 GUI 应用（实时检测、语音控制、声音报警）
- **Windows PC**：运行远程看板（接收事件、监控性能、远程控制）

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
│   └── test_core.py              # pytest 单元测试（67 项）
│
├── tools/                        # 工具（RK3588 板端离线安装包，从 Release 下载）
│   ├── rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
│   │                             # RKNN Lite2 安装包 (559KB, aarch64)
│   └── pyqt5_wheels/             # PyQt5 离线安装包（RK3588 板端专用）
│       ├── install_pyqt5.sh      #   安装脚本
│       ├── python3-pyqt5_5.15.6_arm64.deb
│       ├── python3-pyqt5-sip_12.9.1_arm64.deb
│       ├── python3-sip_4.19.25_arm64.deb
│       └── python3-sip-dev_4.19.25_arm64.deb
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

## 3. 离线依赖包下载

RK3588 板端离线安装包（`.whl`、`.deb`）不随源码分发，从 Release 下载：

1. 前往 [Releases](../../releases/latest) 页面
2. 下载 `tools/` 相关附件
3. 解压到 `tools/` 目录

```
tools/
├── rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
└── pyqt5_wheels/
    ├── install_pyqt5.sh
    ├── python3-pyqt5_5.15.6_arm64.deb
    ├── python3-pyqt5-sip_12.9.1_arm64.deb
    ├── python3-sip_4.19.25_arm64.deb
    └── python3-sip-dev_4.19.25_arm64.deb
```

---

## 4. 快速启动

### RK3588 开发板 — 推理应用

推理应用（GUI）运行在 RK3588 开发板上，用于实时检测：

```bash
cd /home/elf/TBJU_edge_inference_app

# 安装依赖（tools/ 从 Release 下载，见下方说明）
pip install -r requirements.txt
pip install tools/rknn_toolkit_lite2-*.whl
pip install pyserial

# 启动 GUI
export DISPLAY=:0.0
python3 run_gui.py

# 或使用含音频检测的启动脚本
bash start_gui_with_audio.sh

# 冒烟测试
python run_smoke_test.py
```

### Windows PC — 远程看板

远程看板运行在 Windows PC 上，用于接收开发板上传的检测事件和数据：

```bash
cd tbju-dashboard

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

### CSV 监控（Windows PC）

```bash
python watch_csv.py
# 自动监控 output/ 目录，为新 CSV 生成 TXT 摘要
```

---

## 5. GUI 三标签页

| Tab | 名称 | 功能 |
|-----|------|------|
| Tab 1 | 检测识别 | 图片/视频/摄像头检测、语音控制、声音报警、事件上传 |
| Tab 2 | 性能监控 | 实时系统资源监控（CPU/内存/温度/NPU/GPU），历史曲线与表格 |
| Tab 3 | 扩展能力 | 1/2/4 ROI 压力测试，输出容量报告 |

---

## 6. 远程看板 3 Tab 布局

| Tab | 名称 | 内容 |
|-----|------|------|
| Tab1 | 总览 | KPI 卡片 + 告警 + 远程控制 + 日志/命令历史 + 最近事件 + 设备状态 |
| Tab2 | 检测 | 同步文件列表 + 事件表格（7 列）+ 操作栏 |
| Tab3 | 性能 | 6 数值卡片（CPU/内存/温度/推理耗时/NPU/GPU）+ 7 图表 |

---

## 7. CSV 同步流程

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

## 8. 安全特性

- `torch.load(weights_only=True)` 防 pickle 反序列化
- `TBJURKNNEngine._infer_lock` 推理锁（infer_frame / close / reset / load 全部在锁内）
- `EventUploader` 显式 `start()` + 200MB 磁盘限额
- `safe_csv_cell()` 防 CSV 注入
- 看板 `/api/files/download` 路径穿越防护
- SQLite `BEGIN IMMEDIATE` + `busy_timeout=5000`
- 语音报警队列 `maxsize=50`
- 命令 ack 基于回调真实返回值
- 后端 `load()` 失败自动释放已加载资源（try/except → close）
- RKNN `_create_runner()` 局部对象失败时 release

---

## 9. 测试

```bash
# pytest 单元测试（67 项）
python tests/test_core.py

# 冒烟测试（4 项）
python run_smoke_test.py
```

---

## 10. 核心算法：区域约束 + 短时多帧确认

推理后处理流程：

```text
YOLO 推理 → decode_yolov8()
         → validate_debris_region()     ← 区域验证（不改类别）
         → OCR（车号区域）
         → TemporalConsistencyFilter    ← 滑动窗口内命中 M 次确认
             .update_and_filter()
         → draw_detections()
```

### 区域验证 `validate_debris_region()`

- 异物落在对应区域内 → 通过（保留原始类别）
- 对应区域未检测到 → 通过（保守放行，避免漏报）
- 对应区域检测到但异物不在其中 → 过滤（背景误报）
- 按类名解析映射（`DEBRIS_REGION_NAMES`），不依赖固定类别 ID

### 时序一致性 `TemporalConsistencyFilter`

- 滑动窗口 5 帧，至少出现 3 次才确认告警
- 按 `source` 隔离历史（多路视频互不干扰）
- 贪心一对一匹配（同一历史框不会被多个当前框重复匹配）
- 匹配阈值按框对角线比例计算（适配不同分辨率），设有最小像素下限
- 每帧都推进历史（包括空帧），确保旧检测被滑出窗口

### 类名校验

- `load_classes()` 支持乱序显式 ID，校验重复 ID、缺号、格式混用、重复类名
- `resolve_class_ids()` 启动时校验所有必需类别存在
- `tbju_class_id` 按名称解析，不依赖默认值 0

---

## 11. 依赖

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

## 12. 文档索引

| 文档 | 内容 |
|------|------|
| `docs/README_GUI_DEMO.md` | GUI 使用说明（图片/视频/摄像头/语音/报警/上传） |
| `docs/BOARD_RUN_CHECKLIST.md` | ELF2/RK3588 板端部署清单 |
| `docs/LD3320_VOICE_CONTROL.md` | LD3320 语音控制详细文档 |
| `docs/LD3320_STC11_WINDOWS_DEBUG.md` | Windows 串口调试文档 |

---

## 13. 相关项目

- [Carriage_training_pipeline](../Carriage_training_pipeline/) — 模型训练管道
- [TBJU-carriage-detection](../../TBJU-carriage-detection/) — GitHub 仓库主页
