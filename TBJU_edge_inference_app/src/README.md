# 源代码目录

本目录包含推理应用的核心源代码模块。

---

## 目录结构

```
src/
├── __init__.py                        # 包初始化
├── core/                              # 推理核心
│   ├── __init__.py
│   └── tbju_rknn_core.py             # RKNN/PyTorch/ONNX 三后端推理引擎
├── gui/                               # GUI 界面
│   ├── __init__.py
│   ├── tbju_demo_gui.py              # PyQt5 主界面（三标签页）
│   └── tbju_demo_tk.py               # tkinter 回退界面
├── alarm/                             # 声音报警
│   ├── __init__.py
│   └── voice_alarm.py                # 报警播放（队列 + 冷却间隔）
├── network/                           # 网络通信
│   ├── __init__.py
│   ├── utils.py                      # 公共工具（IP 获取、URL 校验）
│   ├── event_uploader.py             # 事件上传（离线队列 + 磁盘持久化）
│   └── command_poller.py             # 远程命令轮询
├── voice/                             # 语音控制
│   ├── __init__.py
│   └── ld3320_uart.py                # LD3320 UART 串口通信
├── monitor/                           # 系统监控
│   ├── __init__.py
│   └── tbju_system_monitor.py        # CPU/GPU/NPU 实时监控
└── capacity/                          # 压力测试
    ├── __init__.py
    └── tbju_capacity_test.py         # 1/2/4 ROI 压测
```

---

## 模块说明

### core — 推理核心

`tbju_rknn_core.py` 是推理引擎核心，包含：

- **后端工厂**：`create_backend()` 自动选择 RKNN/PyTorch/ONNX
- **推理引擎**：`TBJURKNNEngine` 封装模型加载、推理、结果处理
- **后处理**：`decode_yolov8()` 解码、`validate_debris_region()` 区域验证
- **时序滤波**：`TemporalConsistencyFilter` 滑动窗口确认

### gui — GUI 界面

- `tbju_demo_gui.py`：PyQt5 主界面，三标签页布局
  - Tab1：检测识别（图片/视频/摄像头/语音/报警）
  - Tab2：性能监控（CPU/内存/温度曲线）
  - Tab3：扩展能力（1/2/4 ROI 压测）
- `tbju_demo_tk.py`：tkinter 回退界面（PyQt5 不可用时）

### alarm — 声音报警

`voice_alarm.py` 实现异物检测时的语音播报：
- 队列机制（maxsize=50）
- 冷却间隔避免重复播报
- 支持文件播放和 TTS 回退

### network — 网络通信

- `event_uploader.py`：事件上传到远程看板
  - 显式 `start()` 启动后台线程
  - 磁盘持久化离线队列（200MB 限额）
  - 断网不丢事件，网络恢复自动补传
- `command_poller.py`：轮询远程命令
  - KeyError 保护
  - ack 基于回调返回值

### voice — 语音控制

`ld3320_uart.py` 实现 LD3320 语音模块控制：
- UART 串口通信
- 命令映射（中文文本 / hex 匹配）
- 支持动作：打开摄像头、开始/停止检测、录制等

### monitor — 系统监控

`tbju_system_monitor.py` 实现实时系统监控：
- 线程安全采集
- CPU/GPU/NPU 三源数据
- 历史曲线和 CSV 日志

### capacity — 压力测试

`tbju_capacity_test.py` 实现多路检测压力测试：
- 1/2/4 ROI 模式
- 流式 CSV 输出
- 容量报告生成
