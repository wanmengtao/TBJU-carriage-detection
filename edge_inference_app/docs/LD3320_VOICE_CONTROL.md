# LD3320 UART 语音控制接线与调试说明

本项目已加入 LD3320 UART 语音控制支持。它用于“语音命令输入”，不是声音播放模块。没有喇叭时也可以先做语音控制。

## 1. 硬件定位

LD3320 模块适合识别固定命令词，例如：

- 打开摄像头
- 开始检测
- 停止检测
- 暂停检测
- 继续检测
- 开始评估
- 停止评估
- 开始录制 / 停止录制
- 系统状态
- 静音 / 解除静音

识别到命令后，真正通过 UART 发给 ELF2 的内容取决于 STC11 单片机里烧录的程序。官方示例程序默认常见行为是打印中文提示，或向 SYN6288 语音播放模块发送 `FD ...` 播报帧；这类播报帧不是 GUI 控制命令。若要稳定控制本项目，推荐修改 STC 程序，让每个识别结果通过 UART 输出一个明确的单字节命令，例如 `0x01` 表示界面按钮“打开摄像头”，`0x04` 表示界面里的“开始/继续”。

## 2. 接线原则

ELF2 的 40Pin 信号线是 3.3V 电平，不能直接接 5V TTL 信号。

推荐使用 40Pin 引出的 UART9，不建议占用 UART2 Debug 串口。

逻辑接线：

```text
LD3320 VCC  -> ELF2 3.3V 或 5V
LD3320 GND  -> ELF2 GND
LD3320 TXD  -> ELF2 UART9_RX
LD3320 RXD  -> ELF2 UART9_TX
```

注意：

- TX 接 RX，RX 接 TX。
- 必须共地。
- 如果 LD3320 模块虽然支持 5V 供电，但 UART TX/RX 是 5V TTL，需要加电平转换，不能直连 ELF2。
- 为安全起见，优先让 LD3320 用 3.3V 供电，并确认 UART 输出也是 3.3V。

## 3. 引脚依据

ELF2 引脚复用表中 UART9 相关项包括：

```text
P4_21: UART9_TX_M2，UART9 发送
P4_23: UART9_RX_M2，UART9 接收
```

这里的 `P4_21/P4_23` 是资料表里的连接器/复用标识。实际插杜邦线时，请以 ELF2 官方 40Pin 实物丝印、原理图或配套 40Pin 图为准，找到 UART9_TX、UART9_RX、3.3V/5V 和 GND 对应的物理针脚。

## 4. 板端确认串口

接好模块后，在 ELF2 上执行：

```bash
ls /dev/ttyS* /dev/ttyFIQ* /dev/ttyUSB* 2>/dev/null
dmesg | grep -i tty
```

如果 UART9 已启用，常见设备名可能是：

```text
/dev/ttyS9
```

如果系统里没有 `/dev/ttyS9`，需要根据实际设备名修改 GUI 中的串口，或检查设备树是否启用了 UART9。

## 5. 安装依赖

语音控制只额外需要 `pyserial`：

```bash
pip install pyserial
```

## 6. 命令行测试

先不接硬件，测试命令映射：

```bash
python3 scripts/voice_ld3320_test.py --simulate 打开摄像头
python3 scripts/voice_ld3320_test.py --simulate_hex 01
python3 scripts/voice_ld3320_test.py --simulate 开始检测
python3 scripts/voice_ld3320_test.py --simulate_hex 04
```

接上 LD3320 后监听串口：

```bash
python3 scripts/voice_ld3320_test.py --port /dev/ttyS9 --baudrate 9600
```

如果模块实际波特率是 115200：

```bash
python3 scripts/voice_ld3320_test.py --port /dev/ttyS9 --baudrate 115200
```

## 7. GUI 使用

启动 GUI：

```bash
export DISPLAY=:0.0
python3 run_gui.py
```

在“检测识别”页左侧找到“语音控制”：

1. 勾选“启用”。
2. 串口填写 `/dev/ttyS9`，或实际检测到的串口。
3. 波特率填写 `9600` 或模块说明书指定值。
4. 点击“连接语音”。
5. 说命令词，观察日志里是否出现 `[语音命令]`。

## 8. 命令映射配置

配置文件：

```text
config/voice_commands.json
```

默认兼容两类输出：

- 中文文本，例如 `打开摄像头`、`开始检测`
- 二进制十六进制整包，例如 `hex:01`

如果模块输出不同，只需要修改 `matches`。

例子：

```json
"open_camera": {
  "label": "打开摄像头",
  "matches": ["打开摄像头", "hex:01"]
},
"start_detection": {
  "label": "开始/继续检测",
  "matches": ["开始检测", "继续检测", "hex:04"]
}
```

不要把裸数字 `"1"`、`"01"` 作为普通文本规则长期保留。部分 LD3320+SYN6288 程序会输出类似 `FD 00 19 01 01 [v10][m5][t4] ...` 的播放帧，帧内部也含有 `01` 和 `[v10]`，宽松匹配会误触发。当前正式配置默认忽略 SYN6288 风格的 `FD` 播报帧。

## 9. 按官方手册修改 STC11 程序

官方手册说明，该板的关键词和动作不是在 Linux 端设置，而是在 STC11 程序里设置：

- `LDChip.c` 的 `LD_AsrAddFixed()`：设置识别关键词拼音数组 `sRecog[][]`，以及与关键词一一对应的识别码数组 `pCode[]`。
- `LDChip.h`：设置识别码宏定义，例如 `CODE_DMCS 0x01`。
- `main.c` 的 `User_handle(uint8 dat)`：识别成功后的动作。官方示例里有 `UARTSendByte(dat);//串口识别码（十六进制）`，但默认通常是注释状态。

推荐固件设置方式：

```c
// LDChip.h
#define CODE_CMD             0x00  // 一级口令，官方要求 0x00 不要改
#define CODE_OPEN_CAMERA     0x01  // 打开摄像头
#define CODE_STOP_DETECT     0x02  // 停止
#define CODE_PAUSE           0x03  // 暂停
#define CODE_RESUME          0x04  // 开始/继续
#define CODE_START_CAP       0x05  // 开始评估
#define CODE_STOP_CAP        0x06  // 停止评估
#define CODE_TOGGLE_RECORD   0x07  // 开始/停止录制
#define CODE_SHOW_STATUS     0x08  // 系统状态
#define CODE_MUTE_ALARM      0x09  // 静音报警
#define CODE_UNMUTE_ALARM    0x0A  // 解除静音
```

```c
// LDChip.c / LD_AsrAddFixed()
#define DATE_A 12
#define DATE_B 32

uint8 code sRecog[DATE_A][DATE_B] = {
    "gui dao zhu shou",    // 一级口令：轨道助手
    "da kai she xiang tou",
    "kai shi jian ce",
    "ting zhi jian ce",
    "zan ting jian ce",
    "ji xu jian ce",
    "kai shi ping gu",
    "ting zhi ping gu",
    "kai shi lu zhi",
    "xi tong zhuang tai",
    "jing yin bao jing",
    "jie chu jing yin"
};

uint8 code pCode[DATE_A] = {
    CODE_CMD,
    CODE_OPEN_CAMERA,
    CODE_RESUME,
    CODE_STOP_DETECT,
    CODE_PAUSE,
    CODE_RESUME,
    CODE_START_CAP,
    CODE_STOP_CAP,
    CODE_TOGGLE_RECORD,
    CODE_SHOW_STATUS,
    CODE_MUTE_ALARM,
    CODE_UNMUTE_ALARM
};
```

```c
// main.c / User_handle(uint8 dat)
void User_handle(uint8 dat)
{
    if (0 == dat) {
        G0_flag = ENABLE;  // 一级口令通过
        LED = 0;
        return;
    }

    if (ENABLE == G0_flag) {
        G0_flag = DISABLE;
        LED = 1;
        UARTSendByte(dat); // 关键：向 ELF2 输出单字节识别码
    } else {
        PrintCom("请说出一级口令\r\n");
    }
}
```

烧录这个思路后，ELF2 端应看到真正的一字节命令：

```text
打开摄像头 -> hex=01 -> open_camera
停止检测 -> hex=02 -> stop_detection
暂停检测 -> hex=03 -> pause_detection
开始检测/继续检测 -> hex=04 -> start_detection
开始评估 -> hex=05 -> start_capacity
停止评估 -> hex=06 -> stop_capacity
```

## 10. 当前支持的动作

```text
open_camera       打开摄像头
start_detection   开始/继续检测
stop_detection    停止检测
pause_detection   暂停检测
resume_detection  兼容旧配置，等同开始/继续检测
start_capacity    开始评估
stop_capacity     停止评估
toggle_recording  开始/停止录制
show_status       在日志显示系统状态
mute_alarm        记录静音状态，后续声音预警模块会使用
unmute_alarm      解除静音状态
```

## 11. 常见问题

### 连接时报缺少 pyserial

执行：

```bash
pip install pyserial
```

### 串口打不开

检查设备名：

```bash
ls /dev/ttyS* /dev/ttyUSB*
```

检查权限：

```bash
sudo usermod -aG dialout $USER
```

重新登录后再试。临时测试也可使用：

```bash
sudo chmod 666 /dev/ttyS9
```

### 有数据但命令未匹配

查看 GUI 日志或命令行测试输出的 `raw` 和 `hex`，把对应文本或 `hex:xx` 加到 `config/voice_commands.json` 的 `matches` 中。

### 出现 `FD 00 ... [v10][m5][t4] ...`

这通常是 SYN6288 风格的语音播报帧，表示 STC 程序正在让播放模块说一句话，例如“你好主人”或“请说出一级口令”。它不是识别命令。当前测试脚本会显示 `FRAME type=syn6288_tts`，并默认 `NO MATCH`。

如果暂时不能重新烧录 STC11 固件，也可以作为临时调试方案：把 `config/voice_commands.json` 里的 `protocol.ignore_syn6288_tts` 改为 `false`，再把播报帧解码出的文本关键词加入某个动作的 `matches`。这只能用于临时排查串口，不建议用于正式展示；正式方案仍建议让 STC11 输出明确的单字节识别码。

### 识别误触发

当前 GUI 对相同动作做了 1 秒防抖。如果模块误触发频繁，建议在 LD3320 侧减少相似命令词，或把软件防抖时间调大。
