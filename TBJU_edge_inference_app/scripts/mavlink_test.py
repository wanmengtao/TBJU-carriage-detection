#!/usr/bin/env python3
"""
MAVLink 飞控数据接收测试脚本

用法:
  # 连接飞控并实时打印数据
  python3 scripts/mavlink_test.py --port /dev/ttyS4 --baudrate 57600

  # USB 转串口
  python3 scripts/mavlink_test.py --port /dev/ttyUSB0 --baudrate 57600

  # 指定 PX4 系统 ID
  python3 scripts/mavlink_test.py --port /dev/ttyS4 --target-system 1
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.flight.mavlink_receiver import MAVLinkReceiver


def on_status(data):
    """状态回调：实时打印"""
    print(
        f'[{data.timestamp}] '
        f'mode={data.flight_mode:<12s} '
        f'armed={data.armed!s:<5s} '
        f'R={data.roll_deg:+6.1f}° P={data.pitch_deg:+6.1f}° Y={data.yaw_deg:+6.1f}° '
        f'alt={data.relative_alt_m:5.1f}m '
        f'spd={data.groundspeed:4.1f}m/s '
        f'bat={data.battery_voltage:4.1f}V '
        f'gps={data.gps_fix_type}({data.satellites_visible}sat)'
    )


def on_heartbeat(data):
    """心跳回调：连接状态变化"""
    if data.connected:
        print(f'[心跳] 飞控已连接: {data.autopilot_type}, 模式={data.flight_mode}')
    else:
        print('[心跳] 飞控断开')


def main():
    parser = argparse.ArgumentParser(description='MAVLink flight data receiver test')
    parser.add_argument('--port', default='/dev/ttyS4', help='serial port')
    parser.add_argument('--baudrate', type=int, default=57600, help='baudrate')
    parser.add_argument('--target-system', type=int, default=1, help='PX4 system ID')
    parser.add_argument('--duration', type=float, default=0, help='run duration (0=forever)')
    args = parser.parse_args()

    print(f'连接飞控: {args.port} @ {args.baudrate}')
    print(f'目标系统 ID: {args.target_system}')
    print('按 Ctrl+C 停止\n')

    receiver = MAVLinkReceiver(
        port=args.port,
        baudrate=args.baudrate,
        target_system=args.target_system,
    )

    receiver.on_status(on_status)
    receiver.on_heartbeat(on_heartbeat)
    receiver.start()

    start = time.time()
    try:
        while True:
            time.sleep(1)
            if args.duration > 0 and time.time() - start > args.duration:
                break
            # 每秒打印一次紧凑状态
            print(f'  → {receiver.format_compact()}')
    except KeyboardInterrupt:
        print('\n停止中...')
    finally:
        receiver.stop()
        print('已断开')


if __name__ == '__main__':
    main()
