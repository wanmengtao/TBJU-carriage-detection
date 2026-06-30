# TBJU 列车车厢与轨道异物智能检测系统 — 部署包

基于 YOLOv8 统一检测模型 (nc=6) + PP-OCR Rec 车号识别模型。
支持 **Windows 11 PC**（开发/测试）和 **ELF2 / RK3588 开发板**（部署/展示）双平台运行。

---

## 目录结构

```
RKNN_deploy_app_T/
├── run_gui.py                    # GUI 启动入口（PyQt5，自动回退 tkinter）
├── run_smoke_test.py             # 冒烟测试入口
├── watch_csv.py                  # CSV 监控 + TXT 摘要自动生成
├── start_gui_with_audio.sh       # 板端启动脚本（含音频设备检测）
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
├── models/                       # 模型文件
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
├── output/                       # 输出目录（自动创建）
│   ├── images/                   # 图片推理结果
│   ├── videos/                   # 视频推理结果
│   ├── camera/                   # 摄像头推理结果
│   ├── capacity/                 # 压测报告
│   ├── logs/                     # 系统监控日志
│   └── network/events/           # 事件上传队列
│
└── _backup/                      # 旧版本备份
```

---

## 快速启动

### Windows PC

```bash
pip install ultralytics torch opencv-python numpy PyQt5 psutil onnxruntime
cd RKNN_deploy_app_T
python run_gui.py
```

### RK3588 开发板

```bash
cd /home/elf/RKNN_deploy_app_T
pip install tools/rknn_toolkit_lite2-*.whl
pip install opencv-python numpy PyQt5 psutil pyserial
export DISPLAY=:0.0
python3 run_gui.py
```

### 远程看板

```bash
cd tbju-dashboard
pip install fastapi uvicorn python-multipart
python app.py
# 访问 http://localhost:8000
```

### CSV 监控

```bash
python watch_csv.py
# 自动监控 output/ 目录，为新 CSV 生成 TXT 摘要
```

---

## 远程看板 3 Tab 布局

| Tab | 名称 | 内容 |
|-----|------|------|
| Tab1 | 总览 | KPI 卡片 + 告警 + 远程控制 + 日志/命令历史 + 最近事件 + 设备状态 |
| Tab2 | 检测 | 同步文件列表 + 事件表格（7 列）+ 操作栏 |
| Tab3 | 性能 | 6 数值卡片（CPU/内存/温度/推理耗时/NPU/GPU）+ 7 图表 |

---

## CSV 同步流程

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

## 安全特性

- `torch.load(weights_only=True)` 防 pickle 反序列化
- `TBJURKNNEngine._infer_lock` 推理锁防并发
- `EventUploader` 显式 `start()` + 200MB 磁盘限额
- `safe_csv_cell()` 防 CSV 注入
- 看板 `/api/files/download` 路径穿越防护
- SQLite `BEGIN IMMEDIATE` + `busy_timeout=5000`
- 语音报警队列 `maxsize=50`
- 命令 ack 基于回调真实返回值

---

## 测试

```bash
# pytest 单元测试（27 项）
python tests/test_core.py

# 冒烟测试（4 项）
python run_smoke_test.py
```
