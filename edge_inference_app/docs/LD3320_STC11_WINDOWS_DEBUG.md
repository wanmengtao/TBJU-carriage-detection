# LD3320+STC11 Windows 串口调试与一级口令设置

## 结论先说

一级口令不能在 RK3588/Linux/Python 里直接设置。它写在 LD3320+STC11 模块的 STC11 单片机固件中。

Linux/GUI 端只能接收模块串口发出来的数据，例如：

```text
01 -> 打开摄像头
02 -> 停止检测
03 -> 暂停检测
04 -> 开始/继续检测
```

如果模块当前只输出 `FD 00 ... [v10][m5][t4] 你好主人`，说明它在发语音播报帧，不是在发 GUI 控制码。

## 推荐口令设计

建议把一级口令设置为：

```text
轨道助手
```

对应 LD3320 拼音：

```text
gui dao zhu shou
```

原因：

- 比“小智小智”更贴合轨道检测项目。
- 比“小智小智”更不容易被普通说话误唤醒。
- 评审听起来更像项目定制语音入口。

二级命令建议如下：

| 中文命令 | 拼音 | 串口码 | GUI 动作 |
|---|---|---:|---|
| 打开摄像头 | da kai she xiang tou | `0x01` | 打开摄像头 |
| 开始检测 | kai shi jian ce | `0x04` | 开始/继续检测 |
| 停止检测 | ting zhi jian ce | `0x02` | 停止检测 |
| 暂停检测 | zan ting jian ce | `0x03` | 暂停检测 |
| 继续检测 | ji xu jian ce | `0x04` | 开始/继续检测 |
| 开始评估 | kai shi ping gu | `0x05` | 开始评估 |
| 停止评估 | ting zhi ping gu | `0x06` | 停止评估 |
| 开始录制 | kai shi lu zhi | `0x07` | 开始/停止录制 |
| 系统状态 | xi tong zhuang tai | `0x08` | 显示系统状态 |
| 静音报警 | jing yin bao jing | `0x09` | 静音报警 |
| 解除静音 | jie chu jing yin | `0x0A` | 解除静音 |

## STC11 固件需要改哪里

官方示例工程主要改三个位置：

```text
LDChip.h   定义识别码 CODE_*
LDChip.c   在 LD_AsrAddFixed() 中设置关键词拼音 sRecog[][] 和识别码 pCode[]
main.c     在 User_handle(uint8 dat) 中处理识别结果并发送 UART 字节
```

## LDChip.h 推荐代码

保留官方 `CODE_CMD 0x00` 作为一级口令，不要把它改成别的值。

```c
#define CODE_CMD             0x00  // 一级口令，官方建议 0x00 不修改
#define CODE_OPEN_CAMERA     0x01  // 打开摄像头
#define CODE_STOP_DETECT     0x02  // 停止检测
#define CODE_PAUSE_DETECT    0x03  // 暂停检测
#define CODE_RESUME_DETECT   0x04  // 开始/继续检测
#define CODE_START_CAPACITY  0x05  // 开始评估
#define CODE_STOP_CAPACITY   0x06  // 停止评估
#define CODE_TOGGLE_RECORD   0x07  // 开始录制
#define CODE_SHOW_STATUS     0x08  // 系统状态
#define CODE_MUTE_ALARM      0x09  // 静音报警
#define CODE_UNMUTE_ALARM    0x0A  // 解除静音
```

## LDChip.c 推荐代码

在 `LD_AsrAddFixed()` 中，把关键词数组和识别码数组改成下面这样。`DATE_A` 是命令数量，`DATE_B` 是最长拼音字符串长度，建议留大一点。

```c
#define DATE_A 12
#define DATE_B 32

uint8 code sRecog[DATE_A][DATE_B] = {
    "gui dao zhu shou",
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
    CODE_RESUME_DETECT,
    CODE_STOP_DETECT,
    CODE_PAUSE_DETECT,
    CODE_RESUME_DETECT,
    CODE_START_CAPACITY,
    CODE_STOP_CAPACITY,
    CODE_TOGGLE_RECORD,
    CODE_SHOW_STATUS,
    CODE_MUTE_ALARM,
    CODE_UNMUTE_ALARM
};
```

## main.c 推荐代码

在 `User_handle(uint8 dat)` 中，建议不要再输出“你好主人”这类播报帧。正式控制 GUI 时，只输出明确的单字节命令码。

```c
void User_handle(uint8 dat)
{
    if (dat == CODE_CMD) {
        G0_flag = ENABLE;
        LED = 0;
        return;
    }

    if (G0_flag == ENABLE) {
        G0_flag = DISABLE;
        LED = 1;
        UARTSendByte(dat);  // 关键：只发一个字节给 ELF2
    } else {
        PrintCom("NEED_CMD\r\n");
    }
}
```

如果你想在串口助手里看到一级口令成功，也可以在 `dat == CODE_CMD` 时临时加：

```c
PrintCom("CMD_OK\r\n");
```

正式接 GUI 前建议删掉这类提示，只保留二级命令的 `UARTSendByte(dat)`，这样最稳定。

## Windows 串口助手调试

用 Windows 调试比在 RK3588 板端更直观，推荐先这样做。

1. 模块接到 CH340/CH341 USB-TTL，插到 Windows 电脑。
2. 打开设备管理器，确认 COM 号，例如 `COM5`。
3. 打开串口助手，例如 SSCOM、XCOM 或 STCISP 自带串口工具。
4. 设置：

```text
波特率: 9600
数据位: 8
校验位: None
停止位: 1
显示模式: 勾选 HEX 显示
发送模式: 不重要，主要看接收
```

5. 重新给模块上电，观察启动输出。
6. 说一级口令：

```text
轨道助手
```

7. 立刻说二级口令：

```text
打开摄像头
```

正确现象：

```text
接收区 HEX 显示: 01
```

再说：

```text
轨道助手
开始检测
```

正确现象：

```text
接收区 HEX 显示: 04
```

如果仍然看到：

```text
FD 00 ... [v10][m5][t4] 你好主人
```

说明烧录的仍然是语音播报程序，或者 `User_handle()` 里还在驱动 SYN6288 播报，没有改成单字节输出。

## 烧录流程

1. 用 Keil 打开官方 STC11 示例工程。
2. 按上面的方式修改 `LDChip.h`、`LDChip.c`、`main.c`。
3. 编译生成 `.hex` 文件，通常在 `obj/` 目录下。
4. 打开 STCISP。
5. 单片机型号选择：

```text
STC11L08XE
```

6. 选择 USB-TTL 对应 COM 口。
7. 选择编译出来的 `.hex`。
8. 点击下载/编程。
9. 按手册要求重新给 LD3320 模块上电，或按下载按键。
10. 提示下载成功后，用串口助手按上一节验证。

## 接回 RK3588 板端验证

Windows 串口助手确认能收到 `01/02/03` 后，再接回 ELF2/RK3588：

```bash
cd ~/RKNN_deploy_app_T
python3 scripts/voice_ld3320_test.py --port /dev/ttyUSB0 --baudrate 9600
```

说：

```text
轨道助手
打开摄像头
```

应看到：

```text
RAW text='\x01' hex=01
MATCH action=open_camera label=打开摄像头 rule=hex:01
```

再说：

```text
轨道助手
开始检测
```

应看到：

```text
RAW text='\x04' hex=04
MATCH action=start_detection label=开始/继续检测 rule=hex:04
```

这时再打开 GUI 的“语音控制”，选择 `/dev/ttyUSB0`，波特率 `9600`，即可控制检测流程。
