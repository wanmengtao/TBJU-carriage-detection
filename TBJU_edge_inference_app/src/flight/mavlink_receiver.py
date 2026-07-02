#!/usr/bin/env python3
"""
mavlink_receiver.py — PX4 飞控 MAVLink 数据接收模块

通过串口连接 PX4 飞控，接收姿态、GPS、电池等遥测数据。
线程安全，后台运行，内存占用 < 1MB。

依赖: pip install pymavlink

串口接线（ELF2 40Pin）:
  PX4 Telem TX  →  ELF2 UART_RX
  PX4 Telem RX  →  ELF2 UART_TX
  PX4 Telem GND →  ELF2 GND

注意: UART9 已被 LD3320 语音模块占用，飞控请用其他串口（如 UART4/ttyS4）。
"""

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional


# ============================================================
# 数据类
# ============================================================

@dataclass
class FlightData:
    """飞控遥测数据快照（线程安全读取通过 MAVLinkReceiver.latest()）"""

    # 心跳（连接状态）
    connected: bool = False
    flight_mode: str = ""
    armed: bool = False
    system_id: int = 0
    autopilot_type: str = ""

    # 姿态 (ATTITUDE)
    roll_deg: float = 0.0       # 横滚角 (度)
    pitch_deg: float = 0.0      # 俯仰角 (度)
    yaw_deg: float = 0.0        # 航向角 (度)
    rollspeed: float = 0.0      # 横滚角速度 (rad/s)
    pitchspeed: float = 0.0     # 俯仰角速度 (rad/s)
    yawspeed: float = 0.0       # 航向角速度 (rad/s)

    # GPS (GLOBAL_POSITION_INT)
    lat: float = 0.0            # 纬度 (度)
    lon: float = 0.0            # 经度 (度)
    alt_m: float = 0.0          # 海拔高度 (米)
    relative_alt_m: float = 0.0 # 相对起飞点高度 (米, PX4 直接给)
    gps_fix_type: int = 0       # GPS 定位类型 (0=无, 2=2D, 3=3D, ...)
    satellites_visible: int = 0 # 可见卫星数
    vx: float = 0.0             # 北向速度 (m/s)
    vy: float = 0.0             # 东向速度 (m/s)
    vz: float = 0.0             # 地向速度 (m/s)
    hdg: float = 0.0            # 航向 (度, 0-360)

    # 距起飞点距离（自行计算，非 MAVLink 直接给）
    north_m: float = 0.0        # 南北方向距离 (米, 正=北, 负=南)
    east_m: float = 0.0         # 东西方向距离 (米, 正=东, 负=西)

    # 电池 (SYS_STATUS / BATTERY_STATUS)
    battery_voltage: float = 0.0   # 电压 (V)
    battery_current: float = 0.0   # 电流 (A)
    battery_remaining: float = -1  # 剩余百分比 (-1=未知)
    battery_consumed: float = 0.0  # 已消耗 (mAh)

    # VFR_HUD
    airspeed: float = 0.0       # 空速 (m/s)
    groundspeed: float = 0.0    # 地速 (m/s)
    heading: int = 0            # 航向 (度)
    throttle: float = 0.0       # 油门 (0-100%)

    # 时间戳
    last_heartbeat_ts: float = 0.0
    last_attitude_ts: float = 0.0
    last_gps_ts: float = 0.0
    last_battery_ts: float = 0.0
    timestamp: str = ""


# ArduPilot 飞行模式映射
ARDUPILOT_MODE_MAP = {
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO",
    4: "GUIDED", 5: "LOITER", 6: "RTL", 7: "CIRCLE",
    9: "LAND", 11: "DRIFT", 13: "SPORT", 14: "FLIP",
    15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE",
    18: "THROW", 19: "AVOID_ADSB", 20: "GUIDED_NOGPS",
    21: "SMART_RTL", 22: "FLOWHOLD", 23: "FOLLOW",
    24: "ZIGZAG", 25: "SYSTEMID", 26: "AUTOROTATE",
}


# ============================================================
# 接收器
# ============================================================

class MAVLinkReceiver:
    """
    MAVLink 后台接收器。

    用法:
        receiver = MAVLinkReceiver(port='/dev/ttyS4', baudrate=57600)
        receiver.start()

        # 获取最新数据（线程安全快照）
        data = receiver.latest()
        print(f'姿态: roll={data.roll_deg:.1f} pitch={data.pitch_deg:.1f}')

        # 或注册回调
        receiver.on_attitude(lambda d: print(f'roll={d.roll_deg:.1f}'))

        receiver.stop()
    """

    def __init__(
        self,
        port: str = '/dev/ttyS4',
        baudrate: int = 57600,
        source_system: int = 255,     # GCS system ID
        source_component: int = 0,
        target_system: int = 1,       # PX4 system ID (默认 1)
        heartbeat_timeout: float = 5.0,
        request_message_interval: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.source_system = source_system
        self.source_component = source_component
        self.target_system = target_system
        self.heartbeat_timeout = heartbeat_timeout
        self.request_message_interval = request_message_interval

        self._lock = threading.Lock()
        self._latest = FlightData()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._conn = None

        # 起飞点记录（首次 GPS 定位时自动设置）
        self._home_lat: Optional[float] = None
        self._home_lon: Optional[float] = None

        # 回调
        self._on_attitude: Optional[Callable] = None
        self._on_gps: Optional[Callable] = None
        self._on_battery: Optional[Callable] = None
        self._on_heartbeat: Optional[Callable] = None
        self._on_status: Optional[Callable] = None

    # ── 回调注册 ──

    def on_attitude(self, callback: Callable[[FlightData], None]):
        """注册姿态数据回调"""
        self._on_attitude = callback

    def on_gps(self, callback: Callable[[FlightData], None]):
        """注册 GPS 数据回调"""
        self._on_gps = callback

    def on_battery(self, callback: Callable[[FlightData], None]):
        """注册电池数据回调"""
        self._on_battery = callback

    def on_heartbeat(self, callback: Callable[[FlightData], None]):
        """注册心跳回调（连接状态变化时触发）"""
        self._on_heartbeat = callback

    def on_status(self, callback: Callable[[FlightData], None]):
        """注册状态回调（任何数据更新时触发）"""
        self._on_status = callback

    # ── 生命周期 ──

    def start(self):
        """启动后台接收线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name='MAVLinkReceiver', daemon=True,
        )
        self._thread.start()

    def stop(self):
        """停止接收"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._disconnect()

    def close(self):
        """别名，兼容上下文管理器"""
        self.stop()

    def latest(self) -> FlightData:
        """获取最新数据快照（线程安全，返回副本防止读到半新半旧数据）"""
        with self._lock:
            return replace(self._latest)

    def is_connected(self) -> bool:
        """飞控是否在线（心跳超时内收到过心跳）"""
        with self._lock:
            if not self._latest.connected:
                return False
            return (time.time() - self._latest.last_heartbeat_ts) < self.heartbeat_timeout

    def latest_if_connected(self) -> Optional[FlightData]:
        """原子返回最新快照（仅在连接状态有效时），消除 is_connected + latest 的竞态"""
        with self._lock:
            if not self._latest.connected:
                return None
            if time.time() - self._latest.last_heartbeat_ts > self.heartbeat_timeout:
                return None
            return replace(self._latest)

    def reset_home(self):
        """重置起飞点（下次 GPS 定位时重新记录）"""
        with self._lock:
            self._home_lat = None
            self._home_lon = None
        print('[MAVLink] 起飞点已重置')

    # ── 内部实现 ──

    def _connect(self):
        """建立 MAVLink 连接"""
        try:
            from pymavlink import mavutil
            # 修复：直接传 port 和 baud，不要拼接成 "port,baud" 字符串
            self._conn = mavutil.mavlink_connection(
                self.port,
                baud=self.baudrate,
                source_system=self.source_system,
                source_component=self.source_component,
            )
            # 修复：等待目标 system 的心跳，避免先连到非目标设备
            # 排除 MAV_AUTOPILOT_INVALID（非飞控组件的心跳）
            deadline = time.time() + 5
            while time.time() < deadline:
                msg = self._conn.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
                if msg and (not self.target_system or msg.get_srcSystem() == self.target_system):
                    if getattr(msg, 'autopilot', None) == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                        continue
                    self._conn.target_system = msg.get_srcSystem()
                    self._conn.target_component = msg.get_srcComponent()
                    break
            else:
                raise TimeoutError(f'未收到 target_system={self.target_system} 的心跳')
            print(f'[MAVLink] 已连接飞控: {self.port} @ {self.baudrate}')
            print(f'[MAVLink] system_id={self._conn.target_system}, '
                  f'component_id={self._conn.target_component}')

            # 请求消息频率（减少不必要的数据）
            if self.request_message_interval:
                self._request_message_intervals()

            return True
        except ImportError:
            print('[MAVLink] 缺少 pymavlink，请执行: pip install pymavlink')
            self._running = False  # 缺库时停止重连，避免 2 秒循环报错
            return False
        except Exception as e:
            print(f'[MAVLink] 连接失败: {e}')
            return False

    def _disconnect(self):
        """断开连接"""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _request_message_intervals(self):
        """
        请求 PX4 按指定频率发送我们需要的消息。
        减少串口带宽和 CPU 消耗。
        """
        try:
            from pymavlink import mavutil as _mavutil
        except ImportError:
            return

        mav = self._conn.mav
        target = self._conn.target_system
        comp = self._conn.target_component

        # (message_id, interval_us)
        # interval_us=0 表示禁用，1000000=1Hz, 100000=10Hz
        requests = [
            (30, 200000),    # ATTITUDE, 5Hz
            (33, 1000000),   # GLOBAL_POSITION_INT, 1Hz
            (1, 2000000),    # SYS_STATUS (电池), 0.5Hz
            (74, 1000000),   # VFR_HUD, 1Hz
            (24, 1000000),   # GPS_RAW_INT, 1Hz
        ]

        for msg_id, interval in requests:
            try:
                # 修复：补齐 7 个 param + confirmation，使用官方常量
                mav.command_long_send(
                    target, comp,
                    _mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,          # confirmation
                    msg_id,     # param1: message ID
                    interval,   # param2: interval (us)
                    0, 0, 0, 0, # param3-7
                )
            except Exception as e:
                print(f'[MAVLink] 请求消息 {msg_id} 频率失败: {e}')

    def _loop(self):
        """主接收循环（带自动重连）"""
        while self._running:
            if not self._connect():
                self._disconnect()
                # 等待后重试，不要直接退出
                for _ in range(20):  # 2 秒，分段检查 _running
                    if not self._running:
                        return
                    time.sleep(0.1)
                continue

            try:
                while self._running:
                    # 阻塞读取一条消息（超时 1 秒）
                    msg = self._conn.recv_match(blocking=True, timeout=1.0)
                    if msg is None:
                        # 超时，检查心跳是否过期
                        self._check_heartbeat_timeout()
                        continue

                    # 修复：过滤目标 system ID
                    if self.target_system and msg.get_srcSystem() != self.target_system:
                        continue

                    msg_type = msg.get_type()
                    self._dispatch(msg_type, msg)

            except Exception as e:
                if self._running:
                    print(f'[MAVLink] 接收异常，准备重连: {e}')
            finally:
                self._disconnect()
                # 修复：断开时重置数据，避免旧姿态/GPS/电池残留
                with self._lock:
                    old = self._latest
                    self._latest = FlightData(
                        connected=False,
                        last_heartbeat_ts=old.last_heartbeat_ts,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    )
                # 断开时清空起飞点，重连后重新记录
                self._home_lat = None
                self._home_lon = None
                if self._running:
                    time.sleep(1)

    def _check_heartbeat_timeout(self):
        """检查心跳超时"""
        with self._lock:
            if self._latest.connected and self._latest.last_heartbeat_ts > 0:
                if time.time() - self._latest.last_heartbeat_ts > self.heartbeat_timeout:
                    self._latest.connected = False
                    print('[MAVLink] 心跳超时，飞控断开')

    def _dispatch(self, msg_type: str, msg):
        """分发消息到对应处理器（带异常保护，单条消息失败不影响整体）"""
        handlers = {
            'HEARTBEAT': self._handle_heartbeat,
            'ATTITUDE': self._handle_attitude,
            'GLOBAL_POSITION_INT': self._handle_global_position,
            'SYS_STATUS': self._handle_sys_status,
            'BATTERY_STATUS': self._handle_battery_status,
            'VFR_HUD': self._handle_vfr_hud,
            'GPS_RAW_INT': self._handle_gps_raw,
            'STATUSTEXT': self._handle_statustext,
        }
        handler = handlers.get(msg_type)
        if handler:
            try:
                handler(msg)
            except Exception as e:
                print(f'[MAVLink] 处理 {msg_type} 失败: {e}')

    # ── 消息处理器 ──

    def _handle_heartbeat(self, msg):
        now = time.time()
        with self._lock:
            was_connected = self._latest.connected
            self._latest.connected = True
            self._latest.last_heartbeat_ts = now
            self._latest.system_id = msg.get_srcSystem()
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # 修复：使用 pymavlink 常量，PX4=12, ArduPilot=3
            try:
                from pymavlink import mavutil as _mavutil
                AUTOPILOT_PX4 = _mavutil.mavlink.MAV_AUTOPILOT_PX4
                AUTOPILOT_ARDUPILOTMEGA = _mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
            except ImportError:
                AUTOPILOT_PX4 = 12
                AUTOPILOT_ARDUPILOTMEGA = 3

            if msg.autopilot == AUTOPILOT_PX4:
                self._latest.autopilot_type = 'PX4'
                # 修复：flightmode 可能为空/UNKNOWN，加备用解析
                try:
                    mode = self._conn.flightmode
                    if not mode or mode == 'UNKNOWN':
                        mode = _mavutil.mode_string_v10(msg)
                    self._latest.flight_mode = mode
                except Exception:
                    self._latest.flight_mode = f'mode_{msg.custom_mode}'
            elif msg.autopilot == AUTOPILOT_ARDUPILOTMEGA:
                self._latest.autopilot_type = 'ArduPilot'
                self._latest.flight_mode = ARDUPILOT_MODE_MAP.get(
                    msg.custom_mode, f'UNKNOWN({msg.custom_mode})')
            else:
                self._latest.autopilot_type = f'type_{msg.autopilot}'
                self._latest.flight_mode = f'mode_{msg.custom_mode}'

            # 解锁状态
            self._latest.armed = bool(msg.base_mode & 128)  # MAV_MODE_FLAG_SAFETY_ARMED

            data = replace(self._latest)

        if not was_connected:
            print(f'[MAVLink] 飞控已连接: {data.autopilot_type}, '
                  f'模式={data.flight_mode}, '
                  f'{"已解锁" if data.armed else "已锁定"}')

        if self._on_heartbeat:
            self._on_heartbeat(data)

    def _handle_attitude(self, msg):
        now = time.time()
        with self._lock:
            self._latest.roll_deg = math.degrees(msg.roll)
            self._latest.pitch_deg = math.degrees(msg.pitch)
            self._latest.yaw_deg = math.degrees(msg.yaw)
            self._latest.rollspeed = msg.rollspeed
            self._latest.pitchspeed = msg.pitchspeed
            self._latest.yawspeed = msg.yawspeed
            self._latest.last_attitude_ts = now
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            data = replace(self._latest)

        if self._on_attitude:
            self._on_attitude(data)
        if self._on_status:
            self._on_status(data)

    def _handle_global_position(self, msg):
        now = time.time()
        with self._lock:
            self._latest.lat = msg.lat / 1e7
            self._latest.lon = msg.lon / 1e7
            self._latest.alt_m = msg.alt / 1000.0
            self._latest.relative_alt_m = msg.relative_alt / 1000.0
            self._latest.vx = msg.vx / 100.0
            self._latest.vy = msg.vy / 100.0
            self._latest.vz = msg.vz / 100.0
            self._latest.hdg = msg.hdg / 100.0
            self._latest.last_gps_ts = now
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # 计算距起飞点距离
            lat = self._latest.lat
            lon = self._latest.lon
            if (lat != 0 or lon != 0) and self._latest.gps_fix_type >= 2:
                # 首次有效 GPS 定位时记录起飞点（fix_type>=2 表示 2D/3D 定位有效）
                if self._home_lat is None:
                    self._home_lat = lat
                    self._home_lon = lon
                    print(f'[MAVLink] 起飞点已记录: {lat:.6f}, {lon:.6f}')
                # 南北距离: 1 纬度 ≈ 111,320 米
                # 东西距离: 1 经度 ≈ 111,320 * cos(纬度) 米
                self._latest.north_m = (lat - self._home_lat) * 111320.0
                self._latest.east_m = (lon - self._home_lon) * 111320.0 * math.cos(math.radians(lat))

            data = replace(self._latest)

        if self._on_gps:
            self._on_gps(data)
        if self._on_status:
            self._on_status(data)

    def _handle_sys_status(self, msg):
        now = time.time()
        with self._lock:
            # SYS_STATUS 的 voltage_battery 是总电压（mV），保留为权威来源
            self._latest.battery_voltage = msg.voltage_battery / 1000.0
            self._latest.battery_current = msg.current_battery / 100.0
            self._latest.battery_remaining = msg.battery_remaining
            self._latest.last_battery_ts = now
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            data = replace(self._latest)

        if self._on_battery:
            self._on_battery(data)

    def _handle_battery_status(self, msg):
        with self._lock:
            # 修复：BATTERY_STATUS.voltages[0] 是单节电芯电压，不覆盖 SYS_STATUS 的总电压。
            # 只在 SYS_STATUS 未收到时（voltage==0）才使用 BATTERY_STATUS 的电压求和。
            # 过滤 0（未使用电芯）和 65535（无效值）
            if self._latest.battery_voltage <= 0 and msg.voltages:
                valid_voltages = [v for v in msg.voltages if v not in (0, 65535)]
                if valid_voltages:
                    self._latest.battery_voltage = sum(valid_voltages) / 1000.0
            if msg.current_consumed != -1:
                self._latest.battery_consumed = msg.current_consumed
            self._latest.last_battery_ts = time.time()
            data = replace(self._latest)

        if self._on_battery:
            self._on_battery(data)

    def _handle_vfr_hud(self, msg):
        with self._lock:
            self._latest.airspeed = msg.airspeed
            self._latest.groundspeed = msg.groundspeed
            self._latest.heading = msg.heading
            self._latest.throttle = msg.throttle
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            data = replace(self._latest)

        if self._on_status:
            self._on_status(data)

    def _handle_gps_raw(self, msg):
        with self._lock:
            self._latest.gps_fix_type = msg.fix_type
            self._latest.satellites_visible = msg.satellites_visible
            self._latest.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            data = replace(self._latest)

        if self._on_status:
            self._on_status(data)

    def _handle_statustext(self, msg):
        """PX4 状态文本消息（警告/错误）"""
        # 修复：兼容 str 和 bytes，pymavlink 不同版本可能不同
        raw = getattr(msg, 'text', '')
        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode('utf-8', errors='ignore').strip()
        else:
            text = str(raw).strip()

        severity_names = {
            0: 'EMERGENCY', 1: 'ALERT', 2: 'CRITICAL', 3: 'ERROR',
            4: 'WARNING', 5: 'NOTICE', 6: 'INFO', 7: 'DEBUG',
        }
        sev = severity_names.get(getattr(msg, 'severity', -1), 'UNKNOWN')
        print(f'[MAVLink] [{sev}] {text}')

    # ── 工具方法 ──

    def format_display(self) -> str:
        """格式化显示（供 GUI 使用）"""
        d = self.latest()  # 修复：使用 latest() 获取快照，不直接读 _latest
        lines = []

        if not self.is_connected():
            lines.append('飞控: 未连接')
            return '\n'.join(lines)

        lines.append(f'飞控: {d.autopilot_type} ({d.flight_mode})')
        lines.append(f'状态: {"已解锁" if d.armed else "已锁定"}')

        lines.append(f'姿态: R={d.roll_deg:.1f}° P={d.pitch_deg:.1f}° Y={d.yaw_deg:.1f}°')

        if d.relative_alt_m != 0 or d.alt_m != 0:
            lines.append(f'高度: {d.relative_alt_m:.1f}m (海拔 {d.alt_m:.1f}m)')

        if d.north_m != 0 or d.east_m != 0:
            lines.append(f'距起飞点: 北{d.north_m:+.1f}m 东{d.east_m:+.1f}m')

        if d.groundspeed > 0:
            lines.append(f'地速: {d.groundspeed:.1f} m/s')

        if d.lat != 0 or d.lon != 0:
            lines.append(f'GPS: {d.lat:.6f}, {d.lon:.6f}')
            lines.append(f'卫星: {d.satellites_visible}颗 (fix={d.gps_fix_type})')

        if d.battery_voltage > 0:
            bat_str = f'{d.battery_voltage:.1f}V'
            if d.battery_remaining >= 0:
                bat_str += f' ({d.battery_remaining}%)'
            lines.append(f'电池: {bat_str}')

        return '\n'.join(lines)

    def format_compact(self) -> str:
        """紧凑格式（一行显示）"""
        d = self.latest()  # 修复：使用 latest() 获取快照
        if not self.is_connected():
            return '飞控: 未连接'
        parts = [
            d.flight_mode,
            f'R{d.roll_deg:.0f}°P{d.pitch_deg:.0f}°',
            f'alt={d.relative_alt_m:.0f}m',
            f'spd={d.groundspeed:.0f}m/s',
        ]
        if d.battery_voltage > 0:
            parts.append(f'{d.battery_voltage:.0f}V')
        return ' | '.join(parts)
