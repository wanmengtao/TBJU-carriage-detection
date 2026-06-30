# CLI 脚本目录

本目录包含命令行推理和测试脚本。

---

## 脚本列表

| 脚本 | 功能 | 说明 |
|------|------|------|
| `inference_tbju.py` | 单图推理 | 对单张图片执行 YOLO + OCR 检测 |
| `inference_tbju_stream.py` | 视频/摄像头推理 | 流式推理，支持视频文件和摄像头 |
| `inference_tbju_merged.py` | 统一模型推理 | 使用合并模型推理（备份） |
| `voice_alarm_test.py` | 声音报警测试 | 测试报警音频播放 |
| `voice_ld3320_test.py` | LD3320 串口测试 | 测试语音模块通信 |

---

## 使用方法

### 1. inference_tbju.py — 单图推理

```bash
# 基本用法
python scripts/inference_tbju.py --image test.jpg

# 使用所有 NPU 核心
python scripts/inference_tbju.py --image test.jpg --all_cores

# 跳过 OCR（只测 YOLO）
python scripts/inference_tbju.py --image test.jpg --skip_ocr

# 指定输出目录
python scripts/inference_tbju.py --image test.jpg --output results/
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 输入图片路径 | 必填 |
| `--all_cores` | 使用所有 NPU 核心 | 否 |
| `--skip_ocr` | 跳过 OCR 识别 | 否 |
| `--output` | 输出目录 | output/images/ |
| `--conf` | 置信度阈值 | 0.5 |
| `--iou` | IoU 阈值 | 0.45 |

---

### 2. inference_tbju_stream.py — 视频/摄像头推理

```bash
# 视频文件推理
python scripts/inference_tbju_stream.py --video demo.mp4 --output result.avi

# 摄像头实时推理
python scripts/inference_tbju_stream.py --camera /dev/video11 --show --fps 30

# 使用所有 NPU 核心，每 2 帧推理一次
python scripts/inference_tbju_stream.py --video demo.mp4 --every_n 2 --all_cores

# Windows 摄像头
python scripts/inference_tbju_stream.py --camera 0 --show
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video` | 视频文件路径 | — |
| `--camera` | 摄像头设备号或路径 | — |
| `--output` | 输出视频路径 | output/videos/ |
| `--show` | 实时显示预览 | 否 |
| `--fps` | 帧率 | 30 |
| `--every_n` | 每 N 帧推理一次 | 1 |
| `--all_cores` | 使用所有 NPU 核心 | 否 |
| `--skip_ocr` | 跳过 OCR | 否 |

---

### 3. voice_alarm_test.py — 声音报警测试

```bash
# 测试异物报警
python scripts/voice_alarm_test.py --event debris --text "警告，发现异物"

# 测试温度报警
python scripts/voice_alarm_test.py --event temp_high --text "温度过高"

# 等待播放完成
python scripts/voice_alarm_test.py --event debris --text "警告" --wait 5
```

---

### 4. voice_ld3320_test.py — LD3320 串口测试

```bash
# 模拟命令（不接硬件）
python scripts/voice_ld3320_test.py --simulate 打开摄像头
python scripts/voice_ld3320_test.py --simulate_hex 01

# 连接硬件测试
python scripts/voice_ld3320_test.py --port /dev/ttyS9 --baudrate 9600

# Windows 串口
python scripts/voice_ld3320_test.py --port COM5 --baudrate 9600
```

---

## 输出目录

推理结果保存在 `output/` 目录：

```
output/
├── images/<session_id>/               # 图片推理结果
│   ├── *_result.jpg                   #   标注后的图片
│   └── result.csv                     #   检测结果 CSV
├── videos/<session_id>/               # 视频推理结果
│   ├── record.mp4                     #   标注后的视频
│   └── result.csv                     #   检测结果 CSV
└── camera/<session_id>/               # 摄像头推理结果
    ├── record.mp4                     #   录制视频
    └── result.csv                     #   检测结果 CSV
```
