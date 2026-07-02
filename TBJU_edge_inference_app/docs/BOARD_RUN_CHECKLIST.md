# ELF2 / RK3588 板端部署清单

## 1. 上传文件到板端

```bash
scp -r /path/to/TBJU_edge_inference_app/ elf@<板端IP>:/home/elf/TBJU_edge_inference_app/
```

板端目标目录: `/home/elf/TBJU_edge_inference_app/`

必须文件:

```
/home/elf/TBJU_edge_inference_app/
├── run_gui.py
├── run_smoke_test.py
├── start_gui_with_audio.sh
├── README.md
├── src/                          # 全部源码（含 alarm/、network/）
├── scripts/                      # CLI 脚本
├── models/yolo/merged_yolov8.rknn
├── models/yolo/merged_yolov8_fp.rknn
├── models/yolo/merged_yolov8_i8.rknn
├── models/ocr/rec_tbju.rknn
├── models/ocr/rec_tbju_fp.rknn
├── config/merged_classes.txt
├── config/ppocr_keys_v1.txt
├── config/voice_commands.json
├── tests/smoke_core.py
├── assets/audio/                 # 声音报警音频文件
├── docs/LD3320_VOICE_CONTROL.md
└── tools/rknn_toolkit_lite2-*.whl
```

## 2. 安装依赖

```bash
cd /home/elf/TBJU_edge_inference_app

# RKNN Lite2
pip install tools/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 其他依赖
pip install opencv-python numpy PyQt5 psutil pyserial
```

## 3. 依赖检查

```bash
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
python3 -c "import cv2; print('cv2 OK')"
python3 -c "import PyQt5; print('PyQt5 OK')"
python3 -c "import psutil; print('psutil OK')"
python3 -c "import serial; print('pyserial OK')"
```

## 4. 运行冒烟测试

```bash
python3 run_smoke_test.py
```

## 5. 启动 GUI

```bash
export DISPLAY=:0.0
python3 run_gui.py
```

或使用含音频检测的启动脚本：

```bash
bash start_gui_with_audio.sh
```

## 6. CLI 测试

```bash
# 单图
python3 scripts/inference_tbju.py --image test.jpg --all_cores

# 只测 YOLO
python3 scripts/inference_tbju.py --image test.jpg --skip_ocr --all_cores

# 视频
python3 scripts/inference_tbju_stream.py --video demo.mp4 --output output/result.avi --all_cores

# 摄像头
export DISPLAY=:0.0
python3 scripts/inference_tbju_stream.py --camera /dev/video21 --show --fps 30 --every_n 2 --all_cores
```

## 7. LD3320 语音控制测试

```bash
# 不接硬件，先测试配置映射
python3 scripts/voice_ld3320_test.py --simulate 打开摄像头
python3 scripts/voice_ld3320_test.py --simulate_hex 01
python3 scripts/voice_ld3320_test.py --simulate 开始检测
python3 scripts/voice_ld3320_test.py --simulate_hex 04

# 接硬件后，确认串口
ls /dev/ttyS* /dev/ttyFIQ* /dev/ttyUSB* 2>/dev/null
dmesg | grep -i tty

# 默认按 UART9 测试；实际设备名以板端输出为准
python3 scripts/voice_ld3320_test.py --port /dev/ttyS9 --baudrate 9600
```

详细接线见 `docs/LD3320_VOICE_CONTROL.md`。

## 8. 声音报警测试

```bash
# 确认声卡设备
aplay -l

# 测试音频播放
python3 scripts/voice_alarm_test.py --event debris --text "警告，发现异物" --wait 5

# 如果无声，检查：
# 1. USB 小音箱是否插入
# 2. assets/audio/ 下是否有对应 wav 文件
# 3. aplay assets/audio/alarm_test.wav 是否能播放
```

如无中文 TTS，提前录制 `.wav` 文件放入 `assets/audio/` 目录。

## 9. 事件上传与远程看板测试

在 PC 端启动看板服务：

```bash
cd tbju-dashboard
pip install -r requirements.txt
python app.py
```

在板端 GUI 中：

1. 服务地址栏填写 `http://PC端IP:8000/api/events`
2. 勾选"启用上传"
3. 点击"测试连接"
4. PC 端看板出现 `network_test` 事件即表示联动成功

联调前提：电脑和板端在同一网络，PC 防火墙放行 8000 端口。

## 10. 摄像头调试

```bash
ls /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video21 --list-formats-ext

# GStreamer 预览
gst-launch-1.0 v4l2src device=/dev/video21 ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! autovideosink
```

## 11. PX4 飞控 MAVLink 测试

```bash
# 安装依赖
pip install pymavlink

# CLI 测试（先不接 GUI）
python3 scripts/mavlink_test.py --port /dev/ttyS4 --baudrate 57600 --target-system 1

# 验证项：
# - 连接成功（心跳已连接）
# - 模式正确（MANUAL/POSCTL/AUTO，不是 UNKNOWN）
# - 姿态变化（摇动飞控）
# - 电池总压（如 16.4V，不是 4.1V）
# - GPS 坐标更新
# - 断线后自动重连
```

串口接线（ELF2 40Pin）：
```
PX4 Telem TX  → ELF2 UART4_RX
PX4 Telem RX  → ELF2 UART4_TX
PX4 Telem GND → ELF2 GND
```

注意：UART9 已被 LD3320 语音模块占用，飞控请用 UART4。

禁用飞控模块：
```bash
TBJU_DISABLE_FLIGHT=1 python3 run_gui.py
```

## 12. 常见问题

| 问题 | 解决 |
|------|------|
| No module named rknnlite | pip install tools/rknn_toolkit_lite2-*.whl |
| No module named serial | pip install pyserial |
| LD3320 串口打不开 | 检查 /dev/ttyS* 是否存在，必要时 sudo chmod 666 /dev/ttyS9 |
| 摄像头打不开 | ls /dev/video*，确认 /dev/video21 存在 |
| 显示不可用 | export DISPLAY=:0.0 |
| OCR 异常 | 确认 data_format='nchw' 未被删除 |
| 温度过高 | every_n=3 或关闭 all_cores |
| PyQt5 不可用 | pip install PyQt5，或用 CLI 脚本 |
| 声音报警无声 | aplay -l 确认声卡，检查 assets/audio/ 文件 |
| 事件上传失败 | 检查网络连通，确认看板服务已启动，防火墙放行 8000 端口 |
| 飞控连不上 | 检查波特率（57600/115200/921600）、接线（TX↔RX 交叉）、串口权限 |
| 飞控模式显示 UNKNOWN | 确认 PX4 版本，CLI 测试看输出 |
| 电池显示 0V | 确认 PX4 发了 SYS_STATUS 消息 |
| 无 pymavlink 模块 | pip install pymavlink |
