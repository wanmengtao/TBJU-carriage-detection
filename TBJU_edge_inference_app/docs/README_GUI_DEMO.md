# GUI 演示软件使用说明

## 依赖检查

```bash
python -c "import cv2; print('cv2 OK')"
python -c "import numpy; print('numpy OK')"
python -c "import PyQt5; print('PyQt5 OK')"
python -c "import psutil; print('psutil OK')"
python -c "import onnxruntime; print('onnxruntime OK')"
# RK3588 板端额外需要:
python -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
python -c "import serial; print('pyserial OK')"
```

## 启动命令

```bash
cd TBJU_edge_inference_app

# 标准启动（PyQt5，自动回退 tkinter）
python run_gui.py

# RK3588 板端启动（含音频设备检测）
export DISPLAY=:0.0
python3 run_gui.py
# 或
bash start_gui_with_audio.sh
```

## GUI 三标签页

GUI 采用三标签页布局：

- **Tab 1 — 检测识别**：图片/视频/摄像头检测、语音控制、声音报警、事件上传。
- **Tab 2 — 性能监控**：实时系统资源监控（CPU/内存/温度/NPU/GPU），历史曲线与表格。
- **Tab 3 — 扩展能力**：1/2/4 ROI 压力测试，输出容量报告。

## 图片模式

1. 点击左侧"打开图片"
2. 选择 JPG/PNG/BMP 图片
3. 自动执行 YOLO + OCR 检测
4. 中央显示标注结果，右侧显示检测统计
5. 结果自动保存到 `output/images/<session_id>/`

## 视频文件模式

1. 点击左侧"打开视频"
2. 选择 MP4/AVI/MKV 视频
3. 点击播放控制中的"开始"启动推理
4. 可调整"帧间隔"控制推理频率
5. 如需保存标注视频，点击"开始录制"
6. 点击"停止"结束处理
7. 结果保存到 `output/videos/<session_id>/`

注意：`打开视频` 只是选择输入源，不等于已经开始推理。需要点击"开始"后才会启动检测线程。

## 摄像头实时模式

1. 点击左侧"打开摄像头"
2. 默认使用 `/dev/video11` (OV13855)，Windows 使用设备 0
3. 系统直接开始实时检测（不需要再点击"开始"）
4. 如需保存演示视频，点击"开始录制"
5. 点击"停止"释放摄像头
6. 结果保存到 `output/camera/<session_id>/`

## 参数调节

- **置信度**: 检测框过滤阈值，越低检测越多
- **IoU**: NMS 去重阈值
- **帧间隔**: 每 N 帧推理一次（视频/摄像头模式）
- **跳过 OCR**: 仅展示 YOLO 检测速度

## 语音控制

LD3320 UART 语音模块可控制 GUI 操作：

1. 勾选"启用"
2. 串口填写 `/dev/ttyS9`（或实际检测到的串口）
3. 波特率填写 `9600` 或模块说明书指定值
4. 点击"连接语音"
5. 说命令词，观察日志里是否出现 `[语音命令]`

支持的语音动作：打开摄像头、开始/停止/暂停检测、开始/停止评估、开始/停止录制、系统状态、静音/解除静音。

安全开关：关闭时只监听不执行，避免误触发。

详细接线见 `docs/LD3320_VOICE_CONTROL.md`。

## 声音报警

检测到异物或系统异常时，自动通过 USB 小音箱语音播报：

1. 勾选"启用"声音报警
2. 点击"测试播报"验证音频链路
3. 检测到异物时自动播报（带冷却间隔）

支持的报警：异物检测、CPU 过载、内存过高、温度偏高/严重。

音频文件放在 `assets/audio/` 目录下，无文件时回退到系统 TTS。

## 事件上传

检测到关键事件时，通过 HTTP POST 上传到远程看板：

1. 在服务地址栏填写看板地址，如 `http://192.168.1.100:8000/api/events`
2. 勾选"启用上传"
3. 点击"测试连接"验证连通
4. 检测到异物/车号/温度异常时自动上传

离线队列持久化到磁盘，断网不丢事件，网络恢复后自动补传。

远程看板详见 `tbju-dashboard/README.md`。

## 系统监控

Tab 2 实时显示:
- CPU 使用率
- 内存使用率
- 温度 (RK3588)
- NPU/GPU 负载 (RK3588, 如系统接口支持)
- 历史曲线与采样表格

推理期间自动生成 `output/logs/metrics_*.csv`

## 扩展能力测试

Tab 3 用于模拟多路检测压力：

- 1 路 ROI：单路轨道/单摄像头
- 2 路 ROI：两路检测压力
- 4 路 ROI：四路轨道/多摄像头

输出报告：`output/capacity/capacity_report.csv`、`.json`、`capacity_summary.md`

## 模型切换

- 下拉框选择不同 YOLO/OCR 模型版本
- FP 版本精度更高，INT8 版本速度更快
- 点击"重新加载模型"切换

## 输出目录结构

```
output/
├── images/<session_id>/
│   ├── *_result.jpg
│   └── result.csv
├── videos/<session_id>/
│   ├── record.mp4 (或 .avi)
│   └── result.csv
├── camera/<session_id>/
│   ├── record.mp4
│   └── result.csv
├── capacity/
│   ├── capacity_report.csv
│   ├── capacity_report.json
│   └── capacity_summary.md
├── logs/
│   └── metrics_*.csv
└── network/events/
    ├── pending/
    ├── sent/
    └── failed/
```

## 常见问题

**Q: 模型加载失败**
A: 检查模型文件是否存在，RK3588 需要 .rknn 格式，PC 需要 .pt 格式

**Q: 摄像头打不开**
A: 确认设备节点 `ls /dev/video*`，检查是否被其他程序占用

**Q: 视频保存失败**
A: 程序会自动尝试 .mp4 → .avi fallback

**Q: PyQt5 不可用**
A: 安装 `pip install PyQt5`，程序会自动回退到 tkinter GUI

**Q: 声音报警无声**
A: `aplay -l` 确认声卡设备，检查 `assets/audio/` 文件存在，点击"测试播报"验证

**Q: 事件上传失败**
A: 检查网络连通，确认看板服务已启动，防火墙放行 8000 端口

**Q: 串口权限不足**
A: `sudo chmod 666 /dev/ttyS9` 或 `sudo usermod -aG dialout $USER`

## 回退到命令行

```bash
# 单张图片
python scripts/inference_tbju.py --image test.jpg --all_cores

# 视频
python scripts/inference_tbju_stream.py --video demo.mp4 --output output/result.avi --all_cores

# 摄像头
python scripts/inference_tbju_stream.py --camera /dev/video11 --show --fps 30 --every_n 2 --all_cores

# 声音报警测试
python scripts/voice_alarm_test.py --event debris --text "警告，发现异物" --wait 5
```
