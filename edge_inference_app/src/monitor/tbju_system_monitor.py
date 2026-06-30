#!/usr/bin/env python3
"""
tbju_system_monitor.py — 系统监控模块
独立于 GUI，每秒采样一次系统状态。
支持 RK3588 sysfs 接口和 PC psutil。
"""

import csv
import glob
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SystemStats:
    timestamp: str = ''
    cpu_percent: float = 0.0
    cpu_per_core: list = field(default_factory=list)
    cpu_freq_mhz: float = 0.0
    cpu_governor: str = ''
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    process_rss_mb: float = 0.0
    temp_zones: dict = field(default_factory=dict)
    npu_load_percent: float = -1.0
    npu_freq_mhz: float = -1.0
    gpu_load_percent: float = -1.0
    gpu_freq_mhz: float = -1.0
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0


def _read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _parse_first_number(s):
    if s is None:
        return None
    import re
    m = re.search(r'[\d.]+', s)
    return float(m.group()) if m else None


class SystemMonitor:
    RK3588_NPU_TOPS = 6.0

    def __init__(self, sample_interval=1.0, process_pid=None):
        self.sample_interval = sample_interval
        self.process_pid = process_pid or os.getpid()
        self._capabilities = {}
        self._running = False
        self._gpu_load_path = None
        self._gpu_freq_path = None
        self._npu_load_path = None
        self._npu_freq_path = None
        self._thread = None
        self._lock = threading.Lock()
        self._latest = SystemStats()
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None
        self._prev_cpu_times = None
        self._data_sources = {}

    def probe(self) -> Dict[str, bool]:
        caps = {}
        log_lines = []
        self._data_sources = {}

        caps['cpu_percent'] = os.path.exists('/proc/stat')
        if caps['cpu_percent']:
            self._data_sources['cpu_percent'] = '/proc/stat'
        caps['memory'] = os.path.exists('/proc/meminfo')
        if caps['memory']:
            self._data_sources['memory'] = '/proc/meminfo'
        cpu_freq_paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq')
        caps['cpu_freq'] = bool(cpu_freq_paths)
        if cpu_freq_paths:
            self._data_sources['cpu_freq'] = cpu_freq_paths[0]
        caps['cpu_governor'] = bool(glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'))

        thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*')
        caps['temperature'] = len(thermal_zones) > 0
        if thermal_zones:
            self._data_sources['temperature'] = thermal_zones[0] + '/temp'

        # NPU 探测 — 顺便缓存路径
        if os.path.exists('/sys/class/rknpu/load'):
            caps['npu_load'] = True
            self._npu_load_path = '/sys/class/rknpu/load'
        if os.path.exists('/sys/class/rknpu/cur_freq'):
            caps['npu_freq'] = True
            self._npu_freq_path = '/sys/class/rknpu/cur_freq'
        if not caps.get('npu_load'):
            for pat in ['/sys/class/devfreq/*npu*/load', '/sys/class/devfreq/fdab0000.npu/load']:
                paths = glob.glob(pat)
                if paths:
                    caps['npu_load'] = True
                    self._npu_load_path = paths[0]
                    break
        if not caps.get('npu_freq'):
            for pat in ['/sys/class/devfreq/*npu*/cur_freq', '/sys/class/devfreq/fdab0000.npu/cur_freq']:
                paths = glob.glob(pat)
                if paths:
                    caps['npu_freq'] = True
                    self._npu_freq_path = paths[0]
                    break
        # debugfs
        caps['npu_debugfs'] = os.path.exists('/sys/kernel/debug/rknpu/load')
        if caps['npu_debugfs'] and not caps.get('npu_load'):
            caps['npu_load'] = True
            self._npu_load_path = '/sys/kernel/debug/rknpu/load'

        if caps.get('npu_load'):
            self._data_sources['npu_load'] = self._npu_load_path
            log_lines.append(f"[探测] NPU: 已找到 ({self._npu_load_path})")
        else:
            log_lines.append('[探测] NPU: 未找到')
        if caps.get('npu_freq'):
            self._data_sources['npu_freq'] = self._npu_freq_path

        # GPU 探测 — 顺便缓存路径
        if os.path.exists('/sys/class/devfreq/fb000000.gpu/load'):
            caps['gpu_load'] = True
            self._gpu_load_path = '/sys/class/devfreq/fb000000.gpu/load'
        if os.path.exists('/sys/class/devfreq/fb000000.gpu/cur_freq'):
            caps['gpu_freq'] = True
            self._gpu_freq_path = '/sys/class/devfreq/fb000000.gpu/cur_freq'
        if not caps.get('gpu_load'):
            for pat in ['/sys/class/devfreq/*gpu*/load', '/sys/class/devfreq/mali*/load']:
                paths = glob.glob(pat)
                if paths:
                    caps['gpu_load'] = True
                    self._gpu_load_path = paths[0]
                    break
        if not caps.get('gpu_freq'):
            for pat in ['/sys/class/devfreq/*gpu*/cur_freq', '/sys/class/devfreq/mali*/cur_freq']:
                paths = glob.glob(pat)
                if paths:
                    caps['gpu_freq'] = True
                    self._gpu_freq_path = paths[0]
                    break

        if caps.get('gpu_load'):
            self._data_sources['gpu_load'] = self._gpu_load_path
            log_lines.append(f"[探测] GPU: 已找到 ({self._gpu_load_path})")
        else:
            log_lines.append('[探测] GPU: 未找到')
        if caps.get('gpu_freq'):
            self._data_sources['gpu_freq'] = self._gpu_freq_path

        try:
            import psutil
            caps['psutil'] = True
            caps['process_rss'] = True
            self._data_sources['process_rss'] = 'psutil.Process.memory_info()'
        except ImportError:
            caps['psutil'] = False
            caps['process_rss'] = os.path.exists(f'/proc/{self.process_pid}/status')
            if caps['process_rss']:
                self._data_sources['process_rss'] = f'/proc/{self.process_pid}/status'

        try:
            import shutil
            shutil.disk_usage('/')
            caps['disk'] = True
            self._data_sources['disk'] = 'shutil.disk_usage()'
        except Exception:
            caps['disk'] = False

        self._capabilities = caps
        self._probe_log = '\n'.join(log_lines)
        return caps

    def get_probe_log(self) -> str:
        """返回 probe() 产生的探测日志"""
        return getattr(self, '_probe_log', '')

    def get_data_sources(self) -> Dict[str, str]:
        """返回每个指标的数据来源路径/方法。"""
        return dict(self._data_sources)

    def estimate_npu_tops(self) -> float:
        """根据 NPU load 百分比估算当前 TOPS 使用量。"""
        s = self._latest
        if s.npu_load_percent < 0:
            return -1.0
        return round(self.RK3588_NPU_TOPS * s.npu_load_percent / 100, 2)

    def sample(self) -> SystemStats:
        stats = SystemStats(timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))

        with self._lock:
            self._sample_cpu(stats)
        self._sample_memory(stats)
        self._sample_temperature(stats)
        self._sample_npu(stats)
        self._sample_gpu(stats)
        self._sample_disk(stats)

        with self._lock:
            self._latest = stats
        return stats

    def latest(self) -> SystemStats:
        with self._lock:
            return self._latest

    def _sample_cpu(self, stats):
        try:
            import psutil
            # interval=0 返回自上次调用以来的使用率
            # 首次调用需要先建立基线，否则返回 0
            if not hasattr(self, '_cpu_ready'):
                psutil.cpu_percent(interval=0)
                self._cpu_ready = True
            stats.cpu_percent = psutil.cpu_percent(interval=0)
            stats.cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
            freq = psutil.cpu_freq()
            if freq:
                stats.cpu_freq_mhz = freq.current
        except ImportError:
            content = _read_file('/proc/stat')
            if content:
                lines = content.split('\n')
                cpu_line = lines[0] if lines else ''
                parts = cpu_line.split()
                if len(parts) >= 5:
                    times = [int(x) for x in parts[1:5]]
                    if self._prev_cpu_times:
                        delta = [t - p for t, p in zip(times, self._prev_cpu_times)]
                        total = sum(delta)
                        if total > 0:
                            stats.cpu_percent = (1 - delta[3] / total) * 100
                    self._prev_cpu_times = times

        paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq')
        if paths:
            freqs = []
            for p in paths[:4]:
                v = _read_file(p)
                if v:
                    freqs.append(int(v) / 1000)
            if freqs:
                stats.cpu_freq_mhz = sum(freqs) / len(freqs)

        governor_paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
        if governor_paths:
            g = _read_file(governor_paths[0])
            if g:
                stats.cpu_governor = g

    def _sample_memory(self, stats):
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats.memory_total_mb = mem.total / 1024**2
            stats.memory_used_mb = mem.used / 1024**2
            stats.memory_percent = mem.percent
            proc = psutil.Process(self.process_pid)
            stats.process_rss_mb = proc.memory_info().rss / 1024**2
        except ImportError:
            content = _read_file('/proc/meminfo')
            if content:
                info = {}
                for line in content.split('\n'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        info[key] = int(val)
                total = info.get('MemTotal', 0)
                avail = info.get('MemAvailable', 0)
                if total > 0:
                    stats.memory_total_mb = total / 1024
                    stats.memory_used_mb = (total - avail) / 1024
                    stats.memory_percent = (1 - avail / total) * 100

    def _sample_temperature(self, stats):
        zones = glob.glob('/sys/class/thermal/thermal_zone*')
        for zone in sorted(zones):
            zone_name = os.path.basename(zone)
            temp_file = os.path.join(zone, 'temp')
            type_file = os.path.join(zone, 'type')
            temp_val = _read_file(temp_file)
            type_val = _read_file(type_file)
            if temp_val:
                temp_c = _parse_first_number(temp_val)
                if temp_c is not None:
                    if temp_c > 200:
                        temp_c /= 1000.0
                    label = type_val if type_val else zone_name
                    stats.temp_zones[label] = temp_c

    def _sample_npu(self, stats):
        # NPU load: 优先用 probe 缓存的路径
        load_str = None
        if self._npu_load_path:
            load_str = _read_file(self._npu_load_path)
        if load_str is None:
            for pattern in ['/sys/class/rknpu/load', '/sys/class/devfreq/*npu*/load', '/sys/class/devfreq/fdab0000.npu/load']:
                paths = glob.glob(pattern)
                if paths:
                    load_str = _read_file(paths[0])
                    if load_str:
                        self._npu_load_path = paths[0]
                        break
        # debugfs 输出格式: "core0: 15%, core1: 22%, core2: 18%"
        if load_str is None:
            debugfs_str = _read_file('/sys/kernel/debug/rknpu/load')
            if debugfs_str and 'core' in debugfs_str:
                import re
                vals = re.findall(r'(\d+)%', debugfs_str)
                if vals:
                    avg = sum(int(x) for x in vals) / len(vals)
                    stats.npu_load_percent = avg
                    load_str = None  # 已处理
                else:
                    load_str = debugfs_str
            elif debugfs_str:
                load_str = debugfs_str
        if load_str:
            v = _parse_first_number(load_str)
            if v is not None:
                stats.npu_load_percent = v

        # NPU freq: 优先用 probe 缓存的路径
        freq_str = None
        if self._npu_freq_path:
            freq_str = _read_file(self._npu_freq_path)
        if freq_str is None:
            for pattern in ['/sys/class/rknpu/cur_freq', '/sys/class/devfreq/*npu*/cur_freq', '/sys/class/devfreq/fdab0000.npu/cur_freq']:
                paths = glob.glob(pattern)
                if paths:
                    freq_str = _read_file(paths[0])
                    if freq_str:
                        self._npu_freq_path = paths[0]
                        break
        if freq_str:
            v = _parse_first_number(freq_str)
            if v is not None:
                stats.npu_freq_mhz = v / 1e6 if v > 1e6 else v

    def _sample_gpu(self, stats):
        # GPU load: 优先用 probe 缓存的路径
        load_str = None
        if self._gpu_load_path:
            load_str = _read_file(self._gpu_load_path)
        if load_str is None:
            for pattern in ['/sys/class/devfreq/*gpu*/load', '/sys/class/devfreq/mali*/load']:
                paths = glob.glob(pattern)
                if paths:
                    load_str = _read_file(paths[0])
                    if load_str:
                        self._gpu_load_path = paths[0]
                        break
        if load_str:
            # 格式: "42@800000000Hz" 或纯数字 "42"
            if '@' in load_str:
                parts = load_str.split('@')
                v = _parse_first_number(parts[0])
                if v is not None:
                    stats.gpu_load_percent = v
                freq_part = parts[1] if len(parts) > 1 else ''
                fv = _parse_first_number(freq_part)
                if fv is not None:
                    stats.gpu_freq_mhz = fv / 1e6 if fv > 1e6 else fv
            else:
                v = _parse_first_number(load_str)
                if v is not None:
                    stats.gpu_load_percent = v

        # GPU freq: 优先用 probe 缓存的路径
        if stats.gpu_freq_mhz < 0:
            freq_str = None
            if self._gpu_freq_path:
                freq_str = _read_file(self._gpu_freq_path)
            if freq_str is None:
                for pattern in ['/sys/class/devfreq/*gpu*/cur_freq', '/sys/class/devfreq/mali*/cur_freq']:
                    paths = glob.glob(pattern)
                    if paths:
                        freq_str = _read_file(paths[0])
                        if freq_str:
                            self._gpu_freq_path = paths[0]
                            break
            if freq_str:
                v = _parse_first_number(freq_str)
                if v is not None:
                    stats.gpu_freq_mhz = v / 1e6 if v > 1e6 else v

    def _sample_disk(self, stats):
        try:
            import shutil
            usage = shutil.disk_usage('/')
            stats.disk_total_gb = usage.total / 1024**3
            stats.disk_used_gb = usage.used / 1024**3
        except Exception:
            pass

    def start(self, csv_path=None):
        if self._running:
            return
        self._running = True
        self.probe()
        if csv_path:
            self._csv_path = Path(csv_path)
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self._csv_path, 'w', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_mb',
                'process_rss_mb', 'temp_max_c', 'npu_load_percent', 'npu_freq_mhz',
                'gpu_load_percent', 'gpu_freq_mhz', 'estimated_tops',
            ])
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            while self._running:
                stats = self.sample()
                if self._csv_writer and self._csv_file and not self._csv_file.closed:
                    try:
                        max_temp = max(stats.temp_zones.values()) if stats.temp_zones else -1
                        npu_tops = round(self.RK3588_NPU_TOPS * stats.npu_load_percent / 100, 2) if stats.npu_load_percent >= 0 else -1
                        self._csv_writer.writerow([
                            stats.timestamp, f'{stats.cpu_percent:.1f}',
                            f'{stats.memory_percent:.1f}', f'{stats.memory_used_mb:.0f}',
                            f'{stats.process_rss_mb:.0f}', f'{max_temp:.1f}' if max_temp > 0 else 'N/A',
                            f'{stats.npu_load_percent:.1f}' if stats.npu_load_percent >= 0 else 'N/A',
                            f'{stats.npu_freq_mhz:.0f}' if stats.npu_freq_mhz >= 0 else 'N/A',
                            f'{stats.gpu_load_percent:.1f}' if stats.gpu_load_percent >= 0 else 'N/A',
                            f'{stats.gpu_freq_mhz:.0f}' if stats.gpu_freq_mhz >= 0 else 'N/A',
                            f'{npu_tops:.2f}' if npu_tops >= 0 else 'N/A',
                        ])
                        self._csv_file.flush()
                    except (ValueError, OSError):
                        pass
                time.sleep(self.sample_interval)
        finally:
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.close()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        # 线程的 finally 块会关闭 CSV 文件，这里只清理引用
        self._csv_file = None
        self._csv_writer = None

    def format_display(self, show_sources=False) -> str:
        s = self._latest
        lines = [f'CPU: {s.cpu_percent:.0f}%']
        if s.cpu_per_core:
            lines.append(f'  各核心: {" ".join(f"{c:.0f}" for c in s.cpu_per_core[:4])}%')
        if s.cpu_freq_mhz > 0:
            lines.append(f'  频率: {s.cpu_freq_mhz:.0f}MHz')
        if s.cpu_governor:
            lines.append(f'  调频: {s.cpu_governor}')

        lines.append(f'内存: {s.memory_used_mb:.0f}/{s.memory_total_mb:.0f}MB ({s.memory_percent:.0f}%)')
        if s.process_rss_mb > 0:
            lines.append(f'进程内存: {s.process_rss_mb:.0f}MB')

        if s.temp_zones:
            for name, temp in sorted(s.temp_zones.items()):
                lines.append(f'{name}: {temp:.1f}C')

        # NPU devfreq 接口在 RK3588 空闲时返回假 100%，不显示
        if s.gpu_load_percent >= 0:
            lines.append(f'GPU: {s.gpu_load_percent:.0f}%')
        if s.gpu_freq_mhz >= 0:
            lines.append(f'GPU freq: {s.gpu_freq_mhz:.0f}MHz')

        if show_sources and self._data_sources:
            lines.append('')
            lines.append('── 数据来源 ──')
            for key, src in sorted(self._data_sources.items()):
                lines.append(f'{key}: {src}')

        return '\n'.join(lines)
