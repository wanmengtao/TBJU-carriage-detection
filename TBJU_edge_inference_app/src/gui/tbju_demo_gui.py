#!/usr/bin/env python3
"""
tbju_demo_gui.py — 列车车厢检测识别系统 GUI v4
工业暗色主题，基于 PyQt5，三 Tab 布局：
  Tab1: 检测识别（原有功能）
  Tab2: 性能评估（实时曲线 + 监控 + 数据来源 + NPU TOPS）
  Tab3: 扩展能力测试（1/2/4 路压测 + 报告导出）
"""

import sys
import time
import re
import base64
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import torch
except (ImportError, OSError):
    pass

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QEvent
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QTextEdit, QGridLayout,
    QSizePolicy, QScrollArea, QMenuBar, QAction, QMessageBox,
    QTabWidget, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QLineEdit,
)

from src.core.tbju_rknn_core import (
    ModelConfig, TBJURKNNEngine, ResultWriter, FrameResult,
    get_class_name,
)
from src.monitor.tbju_system_monitor import SystemMonitor, SystemStats
from src.alarm.voice_alarm import VoiceAlarmManager
from src.network.event_uploader import (
    DEFAULT_DEVICE_ID, DEFAULT_SERVER_URL, EventUploader,
)
from src.network.command_poller import CommandPoller

# ============================================================
# 路径配置
# ============================================================

DEPLOY_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = DEPLOY_DIR.parent
CARTRIDGE_ROOT = PROJECT_ROOT.parent / 'Carriage'

DEFAULT_OUTPUT_DIR = DEPLOY_DIR / 'output'

DEFAULT_YOLO_MODELS = [
    CARTRIDGE_ROOT / 'YOLO_train' / 'run_merged' / 'train' / 'weights' / 'best.pt',
    DEPLOY_DIR / 'models' / 'yolo' / 'merged_yolov8.rknn',
    DEPLOY_DIR / 'models' / 'yolo' / 'merged_yolov8_fp.rknn',
    DEPLOY_DIR / 'models' / 'yolo' / 'merged_yolov8_i8.rknn',
]
DEFAULT_OCR_MODELS = [
    CARTRIDGE_ROOT / 'OCR_train' / 'output' / 'ppocr_rec_carriage_number' / 'best_model.pth',
    DEPLOY_DIR / 'models' / 'ocr' / 'rec_tbju.rknn',
    DEPLOY_DIR / 'models' / 'ocr' / 'rec_tbju_fp.rknn',
]
DEFAULT_CLASSES_FILE = DEPLOY_DIR / 'config' / 'merged_classes.txt'
DEFAULT_DICT_FILE = DEPLOY_DIR / 'config' / 'ppocr_keys_v1.txt'
DEFAULT_VOICE_CONFIG = DEPLOY_DIR / 'config' / 'voice_commands.json'
DEFAULT_ALARM_AUDIO_DIR = DEPLOY_DIR / 'assets' / 'audio'
DEFAULT_EVENT_SERVER_URL = DEFAULT_SERVER_URL

DEBRIS_CLASS_IDS = {2, 4}
DEBRIS_CLASS_NAMES = {'carriage_rim_debris', 'track_intrusion_debris'}
DETECTION_UPLOAD_COOLDOWN_S = 1.0
OCR_UPLOAD_COOLDOWN_S = 1.0
ALARM_DEBRIS_COOLDOWN_S = 3.0
ALARM_TEMP_COOLDOWN_S = 30.0
ALARM_RESOURCE_COOLDOWN_S = 45.0
ALARM_REPEAT_COUNT = 3
ALARM_REPEAT_GAP_S = 0.25
ALARM_TEMP_WARNING_C = 75.0
ALARM_TEMP_CRITICAL_C = 85.0
ALARM_CPU_WARNING_PERCENT = 95.0
ALARM_MEMORY_WARNING_PERCENT = 90.0

# UI 常量
CAMERA_DEFAULT_WIDTH = 640
CAMERA_DEFAULT_HEIGHT = 480
CAMERA_DEFAULT_FPS = 30
THUMBNAIL_MAX_WIDTH = 480
JPEG_QUALITY = 72
VIDEO_FPS_FALLBACK = 25.0
WORKER_WAIT_TIMEOUT_MS = 3000

# 样式常量
STYLE_SUBTLE = 'color: #8b949e; font-size: 12px;'
STYLE_SUCCESS = 'color: #3fb950; font-size: 12px;'
STYLE_WARNING = 'color: #d29922; font-size: 12px;'
STYLE_ERROR = 'color: #f85149; font-size: 12px;'
STYLE_INFO = 'color: #58a6ff; font-size: 12px;'

# 日志上传采样
LOG_UPLOAD_SAMPLE_INTERVAL = 10  # 每 N 条上传一次普通日志
LOG_UPLOAD_MIN_LEVEL = 'WARNING'  # WARNING 及以上始终上传

# ============================================================
# 工业暗色主题
# ============================================================

DARK_STYLE = """
QMainWindow { background-color: #0d1117; }
QWidget { background-color: #0d1117; color: #c9d1d9; font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif; font-size: 13px; }
QGroupBox { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 14px; padding: 14px 8px 8px 8px; font-weight: bold; color: #58a6ff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #58a6ff; font-size: 12px; letter-spacing: 1px; }
QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 8px 16px; color: #c9d1d9; font-weight: bold; font-size: 13px; min-height: 24px; }
QPushButton:hover { background-color: #30363d; border-color: #58a6ff; color: #58a6ff; }
QPushButton:pressed { background-color: #0d1117; }
QPushButton:disabled { background-color: #161b22; color: #484f58; border-color: #21262d; }
QPushButton#btn_start { background-color: #1a3a2a; border-color: #3fb950; color: #3fb950; }
QPushButton#btn_start:hover { background-color: #244a3a; }
QPushButton#btn_pause { background-color: #3a3a1a; border-color: #d29922; color: #d29922; }
QPushButton#btn_pause:hover { background-color: #4a4a2a; }
QPushButton#btn_stop { background-color: #3d1f1f; border-color: #f85149; color: #f85149; }
QPushButton#btn_stop:hover { background-color: #5a2020; }
QPushButton#btn_record { background-color: #3d1f1f; border-color: #f85149; color: #f85149; }
QPushButton#btn_record:hover { background-color: #5a2020; }
QPushButton#btn_record_active { background-color: #1a3a2a; border-color: #3fb950; color: #3fb950; }
QPushButton#btn_export { background-color: #1a2332; border-color: #58a6ff; color: #58a6ff; }
QPushButton#btn_export:hover { background-color: #243044; }
QPushButton#btn_capacity { background-color: #1a2332; border-color: #bc8cff; color: #bc8cff; }
QPushButton#btn_capacity:hover { background-color: #2a1a3a; }
QDoubleSpinBox, QSpinBox { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 6px 8px; color: #c9d1d9; min-height: 22px; font-size: 13px; }
QDoubleSpinBox:hover, QSpinBox:hover { border-color: #58a6ff; }
QLineEdit { background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 6px 8px; color: #c9d1d9; min-height: 22px; font-size: 12px; }
QLineEdit:hover { border-color: #58a6ff; }
QCheckBox { spacing: 6px; color: #c9d1d9; font-size: 13px; }
QCheckBox::indicator { width: 18px; height: 18px; border: 2px solid #30363d; border-radius: 4px; background-color: #21262d; }
QCheckBox::indicator:checked { background-color: #238636; border-color: #238636; }
QLabel#stats_value { color: #f0f6fc; font-size: 15px; line-height: 1.6; }
QLabel#monitor_value { color: #c9d1d9; font-size: 13px; font-family: "Consolas", "Courier New", monospace; line-height: 1.6; }
QLabel#record_indicator { color: #f85149; font-size: 18px; font-weight: bold; }
QLabel#chart_title { color: #58a6ff; font-size: 13px; font-weight: bold; }
QTextEdit { background-color: #0d1117; border: 1px solid #21262d; border-radius: 8px; color: #c9d1d9; font-family: "Consolas", "Courier New", monospace; font-size: 14px; padding: 8px; }
QStatusBar { background-color: #161b22; border-top: 1px solid #21262d; color: #c9d1d9; font-size: 14px; padding: 6px; }
QScrollArea { border: none; background-color: #0d1117; }
QMenuBar { background-color: #161b22; color: #c9d1d9; border-bottom: 1px solid #30363d; font-size: 14px; }
QMenuBar::item:selected { background-color: #30363d; }
QMenu { background-color: #161b22; border: 1px solid #30363d; color: #c9d1d9; }
QMenu::item:selected { background-color: #1a2332; }
QTabWidget::pane { border: 1px solid #30363d; background-color: #0d1117; }
QTabBar::tab { background-color: #161b22; border: 1px solid #30363d; padding: 6px 16px; color: #8b949e; font-size: 12px; }
QTabBar::tab:selected { background-color: #0d1117; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
QTabBar::tab:hover { color: #c9d1d9; }
QTableWidget { background-color: #0d1117; border: 1px solid #21262d; color: #c9d1d9; gridline-color: #21262d; font-size: 13px; }
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #1a2332; }
QHeaderView::section { background-color: #161b22; color: #58a6ff; border: 1px solid #30363d; padding: 6px; font-weight: bold; }
QComboBox { background-color: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; color: #c9d1d9; min-height: 28px; }
QComboBox:hover { border-color: #58a6ff; }
QComboBox QAbstractItemView { background-color: #161b22; border: 1px solid #30363d; color: #c9d1d9; selection-background-color: #1a2332; }
"""


# ============================================================
# 实时曲线 Widget（QPainter 绘制，无第三方依赖）
# ============================================================

class RealtimeChart(QWidget):
    """轻量实时曲线图，用 QPainter 绘制。"""

    COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff', '#f0883e']

    def __init__(self, title='', max_points=120, y_range=(0, 100), parent=None):
        super().__init__(parent)
        self.title = title
        self.max_points = max_points
        self.y_range = y_range
        self.series = {}  # name -> deque
        self.setMinimumHeight(160)
        self.setMinimumWidth(300)

    def add_point(self, name, value):
        if name not in self.series:
            self.series[name] = deque(maxlen=self.max_points)
        self.series[name].append(float(value))
        self.update()

    def clear_data(self):
        self.series.clear()
        self.update()

    def set_y_range(self, lo, hi):
        self.y_range = (lo, hi)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 背景
        painter.fillRect(0, 0, w, h, QColor('#0d1117'))

        margin_l, margin_r, margin_t, margin_b = 55, 15, 30, 25
        chart_x = margin_l
        chart_y = margin_t
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b

        if chart_w < 50 or chart_h < 30:
            return

        # 标题
        painter.setPen(QColor('#58a6ff'))
        painter.setFont(QFont('Segoe UI', 11, QFont.Bold))
        painter.drawText(QRectF(chart_x, 2, chart_w, 24), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        # 网格
        painter.setPen(QPen(QColor('#21262d'), 1))
        for i in range(5):
            yy = chart_y + chart_h * i / 4
            painter.drawLine(int(chart_x), int(yy), int(chart_x + chart_w), int(yy))
            val = self.y_range[1] - (self.y_range[1] - self.y_range[0]) * i / 4
            painter.setPen(QColor('#484f58'))
            painter.setFont(QFont('Consolas', 9))
            painter.drawText(QRectF(0, yy - 10, chart_x - 5, 20), Qt.AlignRight | Qt.AlignVCenter, f'{val:.0f}')
            painter.setPen(QPen(QColor('#21262d'), 1))

        # 绘制曲线
        for idx, (name, data) in enumerate(self.series.items()):
            if not data:
                continue
            color = QColor(self.COLORS[idx % len(self.COLORS)])
            pen = QPen(color, 2)
            painter.setPen(pen)
            points = list(data)
            n = len(points)
            if n < 2:
                continue
            y_lo, y_hi = self.y_range
            y_span = max(y_hi - y_lo, 1)
            prev_x = chart_x
            prev_y = chart_y + chart_h * (1 - (points[0] - y_lo) / y_span)
            for i in range(1, n):
                px = chart_x + chart_w * i / (self.max_points - 1)
                py = chart_y + chart_h * (1 - (points[i] - y_lo) / y_span)
                painter.drawLine(int(prev_x), int(prev_y), int(px), int(py))
                prev_x, prev_y = px, py

            # 图例
            lx = chart_x + 5
            ly = chart_y + chart_h + 5
            painter.setPen(color)
            painter.setFont(QFont('Consolas', 9))
            painter.drawText(int(lx + idx * 120), int(ly), f'{name}: {points[-1]:.1f}')

        painter.end()


# ============================================================
# 录制控制器
# ============================================================

class RecordingController:
    """封装录制开始/停止/写帧逻辑，供 Worker 复用。"""

    def __init__(self):
        self._start_requested = False
        self._stop_requested = False
        self._is_recording = False
        self._writer = None
        self._log_callback = None

    def set_log_callback(self, callback):
        self._log_callback = callback

    def request_start(self):
        self._start_requested = True

    def request_stop(self):
        self._stop_requested = True

    @property
    def is_recording(self):
        return self._is_recording

    def poll(self, writer, video_path, fps, width, height):
        """检查并处理录制请求，应在每帧循环中调用。"""
        if self._start_requested:
            self._start_requested = False
            if not self._is_recording:
                try:
                    writer.open_video(video_path, fps, width, height)
                    self._is_recording = True
                    self._writer = writer
                    if self._log_callback:
                        self._log_callback('录制开始')
                except Exception as e:
                    if self._log_callback:
                        self._log_callback(f'录制启动失败: {e}')
        if self._stop_requested:
            self._stop_requested = False
            if self._is_recording:
                if writer.video_writer:
                    writer.video_writer.release()
                    writer.video_writer = None
                self._is_recording = False
                if self._log_callback:
                    self._log_callback(f'录制已停止: {video_path}')

    def write_frame(self, writer, vis):
        if self._is_recording and writer.video_writer:
            writer.write_frame(vis)

    def cleanup(self, writer):
        if self._is_recording and writer.video_writer:
            try:
                writer.video_writer.release()
            except Exception:
                pass
            writer.video_writer = None
            self._is_recording = False


# ============================================================
# Worker 线程
# ============================================================

class InferenceWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.stop_requested = False
        self.paused = False
        self.mode = None
        self.source = None
        self.every_n = 2
        self.output_dir = None
        self._writer = None
        self._recorder = RecordingController()
        self._recorder.set_log_callback(lambda msg: self.log_message.emit(msg))

    def request_stop(self):
        self.stop_requested = True

    def request_pause(self):
        self.paused = True

    def request_resume(self):
        self.paused = False

    def request_start_recording(self):
        self._recorder.request_start()

    def request_stop_recording(self):
        self._recorder.request_stop()

    def run(self):
        session_dir = ''
        try:
            if self.mode == 'image':
                session_dir = self._run_image()
            elif self.mode == 'video':
                session_dir = self._run_video()
            elif self.mode == 'camera':
                session_dir = self._run_camera()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit(session_dir)

    def _run_image(self):
        image_path = Path(self.source)
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.error.emit(f'图片读取失败: {image_path}')
            return ''
        session_dir = Path(self.output_dir) / 'images' / time.strftime('%Y%m%d_%H%M%S')
        writer = ResultWriter(session_dir)
        writer.open_csv()
        result = self.engine.infer_frame(frame, image_path.name, 0)
        vis = self.engine.draw(frame, result)
        writer.write_rows(result.rows, result.yolo_ms, result.ocr_ms, result.total_ms)
        writer.save_image(vis, f'{image_path.stem}_result.jpg')
        writer.close()
        self.frame_ready.emit(vis, result)
        self.log_message.emit(
            f'检测完成: {len(result.rows)} 个目标, '
            f'YOLO {result.yolo_ms:.1f}ms, OCR {result.ocr_ms:.1f}ms, '
            f'保存到 {session_dir}')
        return str(session_dir)

    def _run_stream(self, cap, source_name, session_dir, fps, width, height,
                    grab_when_paused=False, reconnect_source=None):
        """视频/摄像头共用的流处理循环。"""
        writer = ResultWriter(session_dir)
        writer.open_csv()
        self._writer = writer
        frame_id = 0
        processed = 0
        last_result = None
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            self.error.emit(f'{source_name} 读取首帧失败')
            cap.release()
            writer.close()
            self._writer = None
            return ''
        record_video_path = session_dir / 'record.avi'
        try:
            while not self.stop_requested:
                if self.paused:
                    if grab_when_paused:
                        cap.grab()
                    time.sleep(0.05)
                    continue
                if frame_id % self.every_n == 0:
                    result = self.engine.infer_frame(frame, source_name, frame_id)
                    last_result = result
                    writer.write_rows(result.rows, result.yolo_ms, result.ocr_ms, result.total_ms)
                    processed += 1
                    elapsed = max(time.time() - start, 1e-6)
                    result.fps = processed / elapsed
                    vis = self.engine.draw(frame, result,
                        f'FPS {result.fps:.1f} | YOLO {result.yolo_ms:.1f}ms | OCR {result.ocr_ms:.1f}ms | frame {frame_id}')
                    self.frame_ready.emit(vis, result)
                    ocr_texts = [r.ocr_text for r in result.rows if r.ocr_text]
                    if ocr_texts:
                        self.log_message.emit(f'frame {frame_id}: OCR={",".join(ocr_texts)}; det={len(result.rows)}')
                else:
                    if last_result is not None:
                        vis = self.engine.draw(frame, last_result,
                            f'skip | every_n={self.every_n} | frame {frame_id}')
                    else:
                        vis = frame.copy()
                    self.frame_ready.emit(vis, None)
                self._recorder.poll(writer, record_video_path, fps, width, height)
                self._recorder.write_frame(writer, vis)
                ret, frame = cap.read()
                if not ret:
                    if reconnect_source is not None:
                        reconnected = False
                        for retry in range(5):
                            if self.stop_requested:
                                break
                            time.sleep(0.5)
                            cap.release()
                            cap = cv2.VideoCapture(reconnect_source)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_DEFAULT_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_DEFAULT_HEIGHT)
                                cap.set(cv2.CAP_PROP_FPS, CAMERA_DEFAULT_FPS)
                                ret, frame = cap.read()
                                if ret:
                                    reconnected = True
                                    self.log_message.emit(f'摄像头重连成功 (第{retry+1}次)')
                                    break
                        if not reconnected:
                            self.log_message.emit('摄像头断开，重连失败')
                            break
                    else:
                        break
                frame_id += 1
        finally:
            self._recorder.cleanup(writer)
            cap.release()
            writer.close()
            self._writer = None
            self.log_message.emit(f'{source_name} 处理完成, 保存到 {session_dir}')
        return str(session_dir)

    def _run_video(self):
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            self.error.emit(f'视频打开失败: {self.source}')
            return ''
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:
            fps = VIDEO_FPS_FALLBACK
        session_dir = Path(self.output_dir) / 'videos' / time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(parents=True, exist_ok=True)
        return self._run_stream(cap, Path(self.source).name, session_dir, fps, width, height)

    def _run_camera(self):
        source = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error.emit(f'摄像头打开失败: {self.source}')
            return ''
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_DEFAULT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_DEFAULT_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_DEFAULT_FPS)
        session_dir = Path(self.output_dir) / 'camera' / time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(parents=True, exist_ok=True)
        return self._run_stream(
            cap, 'camera', session_dir,
            float(CAMERA_DEFAULT_FPS), CAMERA_DEFAULT_WIDTH, CAMERA_DEFAULT_HEIGHT,
            grab_when_paused=True, reconnect_source=source,
        )


class ModelLoadWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            engine = TBJURKNNEngine(self.config)
            engine.load()
            self.finished.emit(engine, '')
        except Exception as e:
            self.finished.emit(None, str(e))


class CapacityWorker(QThread):
    progress = pyqtSignal(str)
    metrics = pyqtSignal(dict)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, video_source, profiles, output_dir, monitor, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.video_source = video_source
        self.profiles = profiles
        self.output_dir = output_dir
        self.monitor = monitor
        self._runner = None

    def run(self):
        try:
            from src.capacity.tbju_capacity_test import CapacityTestRunner
            self._runner = CapacityTestRunner(
                self.engine, self.video_source,
                profiles=self.profiles,
                output_dir=self.output_dir,
                monitor=self.monitor,
                progress_callback=lambda msg: self.progress.emit(msg),
                metrics_callback=lambda d: self.metrics.emit(d),
            )
            results = self._runner.run()
            if results:
                session_dir = self._runner._session_dir
                self.finished.emit(str(session_dir))
            else:
                self.finished.emit('')
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit('')

    def request_stop(self):
        if self._runner:
            self._runner.request_stop()


class VoiceSerialWorker(QThread):
    command_received = pyqtSignal(str, str)
    raw_received = pyqtSignal(str)
    status_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, port, baudrate, config_path, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = int(baudrate)
        self.config_path = Path(config_path)
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        reader = None
        try:
            from src.voice.ld3320_uart import (
                LD3320SerialReader, VoiceCommandMapper, decode_payload
            )
            mapper = VoiceCommandMapper.from_file(self.config_path)
            reader = LD3320SerialReader(self.port, self.baudrate).open()
            self.status_message.emit(f'已连接语音模块: {self.port} @ {self.baudrate}')
            while not self.stop_requested:
                data = reader.read_payload()
                if not data:
                    continue
                raw_text = decode_payload(data)
                raw_hex = data.hex(' ')
                # 控制字符（0x00-0x1F）不可打印，统一显示为 hex
                if raw_text and raw_text.isprintable():
                    raw_desc = raw_text
                else:
                    raw_desc = f'hex:{raw_hex}'
                match = mapper.match(data)
                if match:
                    detail = (
                        f'{match.label} | raw="{raw_desc}" | '
                        f'hex={raw_hex} | rule={match.matched_rule}'
                    )
                    self.command_received.emit(match.action, detail)
                else:
                    self.raw_received.emit(f'未匹配语音数据: raw="{raw_desc}" hex={raw_hex}')
        except Exception as e:
            self.error.emit(str(e))
            self.stop_requested = True  # 出错后自动停止，避免错误刷屏
        finally:
            if reader:
                reader.close()
            self.status_message.emit('语音模块已断开')


# ============================================================
# Tab 1: 检测识别页面
# ============================================================

class DetectionPage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(4, 4, 4, 4)

        # 左侧控制区
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(240)
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # 输入源
        input_group = QGroupBox('输入源')
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(4)
        self.btn_open_image = QPushButton('打开图片')
        self.btn_open_image.clicked.connect(self.main_window._open_image)
        input_layout.addWidget(self.btn_open_image)
        self.btn_open_video = QPushButton('打开视频')
        self.btn_open_video.clicked.connect(self.main_window._open_video)
        input_layout.addWidget(self.btn_open_video)
        self.btn_open_camera = QPushButton('打开摄像头')
        self.btn_open_camera.clicked.connect(self.main_window._open_camera)
        input_layout.addWidget(self.btn_open_camera)
        left_layout.addWidget(input_group)

        # 语音控制
        voice_group = QGroupBox('语音控制')
        voice_layout = QGridLayout(voice_group)
        voice_layout.setHorizontalSpacing(6)
        voice_layout.setVerticalSpacing(6)
        self.voice_enable_check = QCheckBox('启用')
        self.voice_enable_check.setChecked(True)
        voice_layout.addWidget(self.voice_enable_check, 0, 0)
        self.voice_status_label = QLabel('未连接')
        self.voice_status_label.setStyleSheet(STYLE_SUBTLE)
        self.voice_status_label.setWordWrap(True)
        voice_layout.addWidget(self.voice_status_label, 0, 1)

        voice_layout.addWidget(QLabel('串口:'), 1, 0)
        self.voice_port_combo = QComboBox()
        self.voice_port_combo.setEditable(True)
        if sys.platform == 'win32':
            self.voice_port_combo.addItems(['COM3', 'COM4', 'COM5', 'COM6'])
        else:
            self.voice_port_combo.addItems(['/dev/ttyS9', '/dev/ttyS4', '/dev/ttyUSB0', '/dev/ttyUSB1'])
        voice_layout.addWidget(self.voice_port_combo, 1, 1)

        voice_layout.addWidget(QLabel('波特率:'), 2, 0)
        self.voice_baud_combo = QComboBox()
        self.voice_baud_combo.setEditable(True)
        self.voice_baud_combo.addItems(['9600', '115200', '57600', '38400'])
        voice_layout.addWidget(self.voice_baud_combo, 2, 1)

        self.btn_voice_connect = QPushButton('连接语音')
        self.btn_voice_connect.clicked.connect(self.main_window._toggle_voice_control)
        voice_layout.addWidget(self.btn_voice_connect, 3, 0, 1, 2)
        self.btn_voice_execute = QPushButton('语音执行: 关闭')
        self.btn_voice_execute.setCheckable(True)
        self.btn_voice_execute.setChecked(False)
        self.btn_voice_execute.setStyleSheet('color: #f85149;')
        self.btn_voice_execute.clicked.connect(self.main_window._toggle_voice_execution)
        voice_layout.addWidget(self.btn_voice_execute, 4, 0, 1, 2)
        self.voice_last_label = QLabel('命令: --')
        self.voice_last_label.setWordWrap(True)
        self.voice_last_label.setStyleSheet(STYLE_INFO)
        voice_layout.addWidget(self.voice_last_label, 5, 0, 1, 2)
        left_layout.addWidget(voice_group)

        # 声音报警
        alarm_group = QGroupBox('声音报警')
        alarm_layout = QGridLayout(alarm_group)
        alarm_layout.setHorizontalSpacing(6)
        alarm_layout.setVerticalSpacing(6)
        self.alarm_enable_check = QCheckBox('启用')
        self.alarm_enable_check.setChecked(True)
        self.alarm_enable_check.toggled.connect(self.main_window._on_alarm_enabled_changed)
        alarm_layout.addWidget(self.alarm_enable_check, 0, 0)
        self.alarm_status_label = QLabel('就绪')
        self.alarm_status_label.setStyleSheet(STYLE_SUCCESS)
        self.alarm_status_label.setWordWrap(True)
        alarm_layout.addWidget(self.alarm_status_label, 0, 1)
        self.btn_alarm_test = QPushButton('测试播报')
        self.btn_alarm_test.clicked.connect(self.main_window._test_voice_alarm)
        alarm_layout.addWidget(self.btn_alarm_test, 1, 0, 1, 2)
        left_layout.addWidget(alarm_group)

        # 联网与事件上传
        upload_group = QGroupBox('联网上传')
        upload_layout = QGridLayout(upload_group)
        upload_layout.setHorizontalSpacing(6)
        upload_layout.setVerticalSpacing(6)
        self.upload_enable_check = QCheckBox('启用')
        self.upload_enable_check.setChecked(False)
        self.upload_enable_check.toggled.connect(self.main_window._on_upload_enabled_changed)
        upload_layout.addWidget(self.upload_enable_check, 0, 0)
        self.upload_status_label = QLabel('未启用')
        self.upload_status_label.setStyleSheet(STYLE_SUBTLE)
        self.upload_status_label.setWordWrap(True)
        upload_layout.addWidget(self.upload_status_label, 0, 1)

        upload_layout.addWidget(QLabel('服务:'), 1, 0)
        self.upload_url_edit = QLineEdit(DEFAULT_EVENT_SERVER_URL)
        self.upload_url_edit.setToolTip('电脑端事件接收接口，例如 http://电脑端IP:8000/api/events')
        self.upload_url_edit.editingFinished.connect(self.main_window._apply_upload_url)
        upload_layout.addWidget(self.upload_url_edit, 1, 1)

        self.btn_upload_test = QPushButton('测试连接')
        self.btn_upload_test.clicked.connect(self.main_window._test_event_upload)
        upload_layout.addWidget(self.btn_upload_test, 2, 0, 1, 2)
        self.upload_queue_label = QLabel('待传 0 | 成功 0 | 失败 0')
        self.upload_queue_label.setWordWrap(True)
        self.upload_queue_label.setStyleSheet(STYLE_INFO)
        upload_layout.addWidget(self.upload_queue_label, 3, 0, 1, 2)
        self.upload_ip_label = QLabel('板端 IP: --')
        self.upload_ip_label.setWordWrap(True)
        self.upload_ip_label.setStyleSheet(STYLE_SUBTLE)
        upload_layout.addWidget(self.upload_ip_label, 4, 0, 1, 2)
        self.remote_enable_check = QCheckBox('远程控制')
        self.remote_enable_check.setChecked(False)
        self.remote_enable_check.toggled.connect(self.main_window._on_remote_enabled_changed)
        upload_layout.addWidget(self.remote_enable_check, 5, 0, 1, 2)
        self.remote_status_label = QLabel('● 未连接')
        self.remote_status_label.setWordWrap(True)
        self.remote_status_label.setStyleSheet(STYLE_SUBTLE)
        upload_layout.addWidget(self.remote_status_label, 6, 0, 1, 2)
        left_layout.addWidget(upload_group)

        # 播放控制
        control_group = QGroupBox('播放控制')
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(4)
        self.btn_start = QPushButton('开始')
        self.btn_start.setObjectName('btn_start')
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.main_window._start)
        control_layout.addWidget(self.btn_start)
        self.btn_pause = QPushButton('暂停')
        self.btn_pause.setObjectName('btn_pause')
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.main_window._pause)
        control_layout.addWidget(self.btn_pause)
        self.btn_stop = QPushButton('停止')
        self.btn_stop.setObjectName('btn_stop')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.main_window._stop)
        control_layout.addWidget(self.btn_stop)
        left_layout.addWidget(control_group)

        # 录制控制
        record_group = QGroupBox('录制控制')
        record_layout = QHBoxLayout(record_group)
        record_layout.setSpacing(4)
        self.btn_record = QPushButton('开始录制')
        self.btn_record.setObjectName('btn_record')
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.main_window._toggle_recording)
        record_layout.addWidget(self.btn_record)
        self.record_indicator = QLabel('⚫ 未录制')
        self.record_indicator.setObjectName('record_indicator')
        record_layout.addWidget(self.record_indicator)
        left_layout.addWidget(record_group)

        # 检测参数
        param_group = QGroupBox('检测参数')
        param_layout = QGridLayout(param_group)
        param_layout.setHorizontalSpacing(8)
        param_layout.setVerticalSpacing(6)
        param_layout.addWidget(QLabel('置信度:'), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        param_layout.addWidget(self.conf_spin, 0, 1)
        param_layout.addWidget(QLabel('IoU:'), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.9)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        param_layout.addWidget(self.iou_spin, 1, 1)
        param_layout.addWidget(QLabel('帧间隔:'), 2, 0)
        self.every_n_spin = QSpinBox()
        self.every_n_spin.setRange(1, 30)
        self.every_n_spin.setValue(2)
        param_layout.addWidget(self.every_n_spin, 2, 1)
        left_layout.addWidget(param_group)

        # 导出
        export_group = QGroupBox('导出')
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(4)
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLabel(str(DEFAULT_OUTPUT_DIR))
        self.output_dir_edit.setWordWrap(True)
        self.output_dir_edit.setStyleSheet('color: #c9d1d9; font-size: 13px;')
        dir_layout.addWidget(self.output_dir_edit, stretch=1)
        btn_browse = QPushButton('浏览')
        btn_browse.clicked.connect(self.main_window._browse_output_dir)
        dir_layout.addWidget(btn_browse)
        export_layout.addLayout(dir_layout)
        fmt_layout = QHBoxLayout()
        self.export_csv_check = QCheckBox('CSV')
        self.export_csv_check.setChecked(True)
        fmt_layout.addWidget(self.export_csv_check)
        self.export_img_check = QCheckBox('图片')
        self.export_img_check.setChecked(True)
        fmt_layout.addWidget(self.export_img_check)
        self.export_video_check = QCheckBox('视频')
        self.export_video_check.setChecked(True)
        fmt_layout.addWidget(self.export_video_check)
        export_layout.addLayout(fmt_layout)
        self.btn_export = QPushButton('导出结果')
        self.btn_export.setObjectName('btn_export')
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.main_window._export_results)
        export_layout.addWidget(self.btn_export)
        left_layout.addWidget(export_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_inner)
        layout.addWidget(left_scroll)

        # 中央画面区
        self.image_label = QLabel('等待打开图片 / 视频 / 摄像头')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            'background-color: #010409; color: #30363d; font-size: 22px; '
            'border: 2px solid #21262d; border-radius: 8px;')
        layout.addWidget(self.image_label, stretch=1)

        # 右侧结果区
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setFixedWidth(260)
        right_inner = QWidget()
        right_layout = QVBoxLayout(right_inner)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(2, 2, 2, 2)

        stats_group = QGroupBox('检测统计')
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel('等待检测...')
        self.stats_label.setObjectName('stats_value')
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignTop)
        stats_layout.addWidget(self.stats_label)
        right_layout.addWidget(stats_group, stretch=1)

        log_group = QGroupBox('日志')
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(80)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, stretch=4)

        monitor_group = QGroupBox('系统监控')
        monitor_layout = QVBoxLayout(monitor_group)
        self.monitor_label = QLabel('CPU: --\n内存: --\n温度: --')
        self.monitor_label.setObjectName('monitor_value')
        monitor_layout.addWidget(self.monitor_label)
        right_layout.addWidget(monitor_group, stretch=2)

        right_scroll.setWidget(right_inner)
        layout.addWidget(right_scroll)

        # 安装事件过滤器：禁用 SpinBox/ComboBox 的滚轮，避免滚动面板时误改参数
        for spin in (self.conf_spin, self.iou_spin, self.every_n_spin):
            spin.installEventFilter(self)
        self.voice_port_combo.installEventFilter(self)
        self.voice_baud_combo.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, (QDoubleSpinBox, QSpinBox, QComboBox)):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


# ============================================================
# Tab 2: 性能评估页面
# ============================================================

class PerformancePage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        # 整页用 QScrollArea 包裹
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 上方：实时状态 + 数据来源（左右分栏）
        top_splitter = QSplitter(Qt.Horizontal)

        # 左：实时数值（带滚动）
        values_widget = QWidget()
        values_layout = QVBoxLayout(values_widget)
        values_group = QGroupBox('实时系统状态')
        vg_layout = QVBoxLayout(values_group)
        self.perf_monitor_label = QLabel('等待采样...')
        self.perf_monitor_label.setObjectName('monitor_value')
        self.perf_monitor_label.setAlignment(Qt.AlignTop)
        self.perf_monitor_label.setWordWrap(True)
        vg_layout.addWidget(self.perf_monitor_label)
        values_layout.addWidget(values_group)

        # GPU 状态（NPU devfreq 接口在 RK3588 空闲时返回假 100%，不显示）
        gpu_group = QGroupBox('GPU 状态')
        gpu_layout = QVBoxLayout(gpu_group)
        self.npu_label = QLabel('GPU: N/A')
        self.npu_label.setObjectName('monitor_value')
        self.npu_label.setWordWrap(True)
        gpu_layout.addWidget(self.npu_label)
        values_layout.addWidget(gpu_group)

        values_layout.addStretch()
        top_splitter.addWidget(values_widget)

        # 右：数据来源
        source_widget = QWidget()
        source_layout = QVBoxLayout(source_widget)
        source_group = QGroupBox('监控数据来源')
        sg_layout = QVBoxLayout(source_group)
        self.source_text = QTextEdit()
        self.source_text.setReadOnly(True)
        self.source_text.setMaximumHeight(200)
        self.source_text.setStyleSheet('font-size: 12px;')
        sg_layout.addWidget(self.source_text)
        source_layout.addWidget(source_group)

        # 历史表格
        history_group = QGroupBox('最近采样记录 (最新 20 条)')
        h_layout = QVBoxLayout(history_group)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            '时间', 'CPU%', '内存MB', '温度C', 'CPU频率MHz'
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setMaximumHeight(180)
        h_layout.addWidget(self.history_table)
        source_layout.addWidget(history_group)

        top_splitter.addWidget(source_widget)
        top_splitter.setSizes([350, 450])
        layout.addWidget(top_splitter, stretch=1)

        # 下方：实时曲线
        charts_group = QGroupBox('实时曲线')
        charts_layout = QHBoxLayout(charts_group)

        self.chart_cpu = RealtimeChart('CPU / 内存 (%)', max_points=120, y_range=(0, 100))
        charts_layout.addWidget(self.chart_cpu)

        self.chart_temp = RealtimeChart('温度 (C)', max_points=120, y_range=(20, 100))
        charts_layout.addWidget(self.chart_temp)

        layout.addWidget(charts_group, stretch=1)

        # 带宽监控
        bw_group = QGroupBox('网络带宽 (开发板 → PC)')
        bw_layout = QHBoxLayout(bw_group)

        self.chart_bandwidth = RealtimeChart('上传速率 (KB/s)', max_points=120, y_range=(0, 100))
        bw_layout.addWidget(self.chart_bandwidth)

        bw_info = QWidget()
        bw_info_layout = QVBoxLayout(bw_info)
        self.bandwidth_label = QLabel('当前: -- KB/s\n峰值: -- KB/s\n累计: -- KB')
        self.bandwidth_label.setObjectName('monitor_value')
        self.bandwidth_label.setWordWrap(True)
        bw_info_layout.addWidget(self.bandwidth_label)
        bw_info_layout.addStretch()
        bw_layout.addWidget(bw_info)

        layout.addWidget(bw_group, stretch=1)

        scroll.setWidget(content)


# ============================================================
# Tab 3: 扩展能力测试页面
# ============================================================

class CapacityPage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 顶部控制区
        ctrl_group = QGroupBox('扩展能力评估')
        ctrl_layout = QVBoxLayout(ctrl_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel('视频源:'))
        self.video_source_edit = QLabel('未选择')
        self.video_source_edit.setStyleSheet('color: #c9d1d9; font-size: 13px;')
        row1.addWidget(self.video_source_edit, stretch=1)
        btn_select_video = QPushButton('选择视频')
        btn_select_video.clicked.connect(self._select_video)
        row1.addWidget(btn_select_video)
        ctrl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_start_capacity = QPushButton('开始评估')
        self.btn_start_capacity.setObjectName('btn_capacity')
        self.btn_start_capacity.clicked.connect(self.main_window._start_capacity_test)
        row2.addWidget(self.btn_start_capacity)
        self.btn_stop_capacity = QPushButton('停止评估')
        self.btn_stop_capacity.setObjectName('btn_stop')
        self.btn_stop_capacity.setEnabled(False)
        self.btn_stop_capacity.clicked.connect(self.main_window._stop_capacity_test)
        row2.addWidget(self.btn_stop_capacity)
        row2.addStretch()
        self.capacity_status = QLabel('就绪')
        self.capacity_status.setStyleSheet('color: #8b949e;')
        row2.addWidget(self.capacity_status)
        ctrl_layout.addLayout(row2)

        # 进度
        self.capacity_progress = QLabel('')
        self.capacity_progress.setStyleSheet('color: #58a6ff; font-size: 14px;')
        ctrl_layout.addWidget(self.capacity_progress)

        layout.addWidget(ctrl_group)

        # 下方：结果表格 + 实时曲线
        bottom_splitter = QSplitter(Qt.Horizontal)

        # 左：实时指标曲线
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        self.capacity_chart = RealtimeChart('压测实时延迟 (ms)', max_points=300, y_range=(0, 500))
        chart_layout.addWidget(self.capacity_chart)
        bottom_splitter.addWidget(chart_widget)

        # 右：结果表格
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_group = QGroupBox('评估结果')
        rg_layout = QVBoxLayout(result_group)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(10)
        self.result_table.setHorizontalHeaderLabels([
            'Profile', '路数', 'every_n', '平均FPS', 'P50ms', 'P95ms',
            'CPU%', '温度C', 'NPU%', '建议'
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.verticalHeader().setVisible(False)
        rg_layout.addWidget(self.result_table)

        # 报告导出按钮
        export_row = QHBoxLayout()
        self.btn_open_report_dir = QPushButton('打开报告目录')
        self.btn_open_report_dir.setEnabled(False)
        self.btn_open_report_dir.clicked.connect(self.main_window._open_report_dir)
        export_row.addWidget(self.btn_open_report_dir)
        export_row.addStretch()
        rg_layout.addLayout(export_row)

        result_layout.addWidget(result_group)
        bottom_splitter.addWidget(result_widget)
        bottom_splitter.setSizes([500, 600])

        layout.addWidget(bottom_splitter, stretch=1)

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择测试视频', '', '视频文件 (*.mp4 *.avi *.mkv *.mov)')
        if path:
            self.video_source_edit.setText(path)
            self.main_window._log(f'压测视频源已选择: {path}')


# ============================================================
# 主窗口
# ============================================================

class MainWindow(QMainWindow):
    remote_command_signal = pyqtSignal(str)
    remote_param_signal = pyqtSignal(dict)
    remote_status_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('TBJU 列车车厢检测识别系统')
        self.setMinimumSize(960, 540)

        self.engine = None
        self.worker = None
        self.capacity_worker = None
        self.voice_worker = None
        self.current_result = None
        self.system_monitor = SystemMonitor()
        self.voice_alarm = VoiceAlarmManager(DEFAULT_ALARM_AUDIO_DIR, enabled=True)
        self.event_uploader = EventUploader(
            DEFAULT_OUTPUT_DIR / 'network' / 'events',
            server_url=DEFAULT_EVENT_SERVER_URL,
            enabled=False,
        )
        self.event_uploader.start()
        self._last_session_dir = None
        self._last_report_dir = None
        self._recording = False
        self._voice_muted = False
        self._last_voice_action = ''
        self._last_voice_action_ts = 0.0
        self._monitor_history = deque(maxlen=20)
        self._system_alarm_counts = {
            'cpu_high': 0,
            'memory_high': 0,
        }
        self._last_upload_by_key = {}
        self._current_pixmap = None
        self._metric_upload_tick = 0
        self._peak_bw_kbps = 0.0
        self._log_counter = 0
        self._sync_timer = None
        self._closing = False

        # 远程控制
        self.command_poller = None
        self.remote_command_signal.connect(self._execute_voice_action)
        self.remote_param_signal.connect(self._apply_remote_params)
        self.remote_status_signal.connect(self._update_remote_status)

        self._init_menu()
        self._init_ui()
        self._init_monitor()
        self._auto_load_model()

    def _init_menu(self):
        menubar = self.menuBar()
        tools_menu = menubar.addMenu('工具')
        switch_action = QAction('切换模型', self)
        switch_action.triggered.connect(self._show_model_switch_dialog)
        tools_menu.addAction(switch_action)
        about_action = QAction('关于', self)
        about_action.triggered.connect(lambda: QMessageBox.about(
            self, '关于', 'TBJU 列车车厢检测识别系统 v4.0\n\n基于 YOLOv8 + PP-OCR Rec\n支持 RK3588 / Windows\n\nTab1: 检测识别\nTab2: 性能评估\nTab3: 扩展能力测试'))
        tools_menu.addAction(about_action)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: 检测识别
        self.detection_page = DetectionPage(self)
        self.tab_widget.addTab(self.detection_page, '检测识别')

        # Tab 2: 性能评估
        self.perf_page = PerformancePage(self)
        self.tab_widget.addTab(self.perf_page, '性能评估')

        # Tab 3: 扩展能力测试
        self.capacity_page = CapacityPage(self)
        self.tab_widget.addTab(self.capacity_page, '扩展能力测试')

        # 状态栏
        self.statusBar().showMessage('就绪')

        # 监控定时器
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._update_monitor)
        self.monitor_timer.start(1000)

        self.upload_status_timer = QTimer()
        self.upload_status_timer.timeout.connect(self._update_upload_status)
        self.upload_status_timer.start(1000)

    def _init_monitor(self):
        caps = self.system_monitor.probe()
        available = [k for k, v in caps.items() if v]
        self._log(f'系统监控: {", ".join(available)}')
        # 填充数据来源
        sources = self.system_monitor.get_data_sources()
        if sources:
            lines = []
            for key, src in sorted(sources.items()):
                lines.append(f'{key}:\n  {src}')
            self.perf_page.source_text.setText('\n\n'.join(lines))

    def _toggle_voice_control(self):
        if self.voice_worker and self.voice_worker.isRunning():
            self._stop_voice_control()
        else:
            self._start_voice_control()

    def _toggle_voice_execution(self, checked):
        dp = self.detection_page
        if checked:
            dp.btn_voice_execute.setText('语音执行: 开启')
            dp.btn_voice_execute.setStyleSheet('color: #3fb950;')
            self._log('[语音] 已开启语音命令执行')
        else:
            dp.btn_voice_execute.setText('语音执行: 关闭')
            dp.btn_voice_execute.setStyleSheet('color: #f85149;')
            self._log('[语音] 已关闭语音命令执行，仅监听不操作')

    def _on_alarm_enabled_changed(self, checked):
        self._voice_muted = not checked
        self.voice_alarm.set_enabled(checked)
        if checked:
            self._set_alarm_status('就绪', '#3fb950')
            self._log('[报警] 声音报警已开启')
        else:
            self._set_alarm_status('已关闭', '#8b949e')
            self._log('[报警] 声音报警已关闭')

    def _set_alarm_status(self, text, color='#d29922'):
        label = getattr(self.detection_page, 'alarm_status_label', None)
        if label:
            label.setText(text)
            label.setStyleSheet(f'color: {color}; font-size: 12px;')

    def _test_voice_alarm(self):
        text = '声音报警测试正常'
        queued = self.voice_alarm.trigger('test', text, cooldown_s=1.0, force=True)
        if queued:
            self._set_alarm_status('测试播报中', '#d29922')
            self._log(f'[报警] {text}')
            self.statusBar().showMessage('声音报警测试播报')
        else:
            self._log('[报警] 播报队列已满，测试播报未加入队列')

    def _on_upload_enabled_changed(self, checked):
        self._apply_upload_url()
        self.event_uploader.set_enabled(checked)
        if checked:
            self._set_upload_status('已启用', '#3fb950')
            self._log('[联网] 事件上传已启用')
        else:
            self._set_upload_status('未启用', '#8b949e')
            self._log('[联网] 事件上传已关闭，检测不会进入上传队列')
        self._update_upload_status()

    def _on_remote_enabled_changed(self, checked):
        if checked:
            self._apply_upload_url()  # 确保 URL 已同步最新值
            base_url = self.event_uploader.server_url.rsplit('/api/events', 1)[0]
            self.command_poller = CommandPoller(dashboard_url=base_url)
            self.command_poller.on_command(lambda action: self.remote_command_signal.emit(action))
            self.command_poller.on_param_change(lambda params: self.remote_param_signal.emit(params))
            self.command_poller.on_status_change(lambda connected: self.remote_status_signal.emit(connected))
            self.command_poller.start()
            self.detection_page.remote_status_label.setText('● 轮询中...')
            self.detection_page.remote_status_label.setStyleSheet('color: #d29922; font-size: 12px;')
            self._log('[远程] 远程控制已启用')
        else:
            if self.command_poller:
                self.command_poller.stop()
                self.command_poller = None
            self.detection_page.remote_status_label.setText('● 未连接')
            self.detection_page.remote_status_label.setStyleSheet(STYLE_SUBTLE)
            self._log('[远程] 远程控制已关闭')

    def _apply_remote_params(self, params):
        dp = self.detection_page
        if 'confidence' in params:
            dp.conf_spin.setValue(float(params['confidence']))
        if 'iou' in params:
            dp.iou_spin.setValue(float(params['iou']))
        if 'frame_interval' in params:
            dp.every_n_spin.setValue(int(params['frame_interval']))
        self._log(f'[远程] 参数已更新: {params}')
        return True

    def _update_remote_status(self, connected):
        if connected:
            self.detection_page.remote_status_label.setText('● 已连接')
            self.detection_page.remote_status_label.setStyleSheet(STYLE_SUCCESS)
        else:
            self.detection_page.remote_status_label.setText('● 连接断开')
            self.detection_page.remote_status_label.setStyleSheet(STYLE_ERROR)

    def _apply_upload_url(self):
        dp = self.detection_page
        url = dp.upload_url_edit.text().strip()
        if not self.event_uploader.set_server_url(url):
            self._log(f'[联网] URL 格式无效，已忽略: {url}')
            self.statusBar().showMessage('URL 格式无效，需要 http:// 或 https:// 开头')

    def _set_upload_status(self, text, color='#d29922'):
        label = getattr(self.detection_page, 'upload_status_label', None)
        if label:
            label.setText(text)
            label.setStyleSheet(f'color: {color}; font-size: 12px;')

    def _test_event_upload(self):
        self._apply_upload_url()
        self._set_upload_status('测试中', '#d29922')
        self.statusBar().showMessage('正在测试事件上传接口...')
        result = self.event_uploader.test_connection()
        if result.get('ok'):
            self._set_upload_status('连接正常', '#3fb950')
            self._log('[联网] 测试事件上传成功')
            self.statusBar().showMessage('事件上传接口正常')
        else:
            err = result.get('error') or str(result.get('response', '未知错误'))
            self._set_upload_status('连接失败', '#f85149')
            self._log(f'[联网 ERROR] 测试事件上传失败: {err}')
            self.statusBar().showMessage('事件上传接口不可用')
        self._update_upload_status()

    def _update_upload_status(self):
        if not hasattr(self, 'detection_page') or not hasattr(self, 'event_uploader'):
            return
        status = self.event_uploader.status()
        dp = self.detection_page
        dp.upload_queue_label.setText(
            f"待传 {status.get('pending_count', 0)} | "
            f"成功 {status.get('success_count', 0)} | "
            f"失败 {status.get('failure_count', 0)}"
        )
        ip = status.get('board_ip') or '--'
        dp.upload_ip_label.setText(f'板端 IP: {ip}')

        if status.get('enabled'):
            if status.get('last_error') and status.get('pending_count', 0) > 0:
                self._set_upload_status('等待重传', '#d29922')
            elif status.get('last_success'):
                self._set_upload_status('上传正常', '#3fb950')
            else:
                self._set_upload_status('已启用', '#3fb950')
        else:
            self._set_upload_status('未启用', '#8b949e')

    def _alarm_enabled(self):
        checkbox = getattr(self.detection_page, 'alarm_enable_check', None)
        return bool(checkbox and checkbox.isChecked() and not self._voice_muted)

    def _trigger_voice_alarm(self, event_key, text, cooldown_s):
        if not self._alarm_enabled():
            return False
        queued = self.voice_alarm.trigger(
            event_key,
            text,
            cooldown_s=cooldown_s,
            repeat_count=ALARM_REPEAT_COUNT,
            repeat_gap_s=ALARM_REPEAT_GAP_S,
        )
        if queued:
            self._set_alarm_status(text, '#d29922')
            self._log(
                f'[报警] {text}（重复 {ALARM_REPEAT_COUNT} 遍，'
                f'同类报警冷却 {cooldown_s:.0f}s）'
            )
            self.statusBar().showMessage(f'声音报警: {text}')
            self._queue_alarm_event(event_key, text)
        return queued

    def _check_detection_alarm(self, result: FrameResult):
        if not result or not result.rows:
            return

        debris_names = set()
        for row in result.rows:
            class_name = row.class_name or get_class_name(row.class_id, self.engine.classes)
            if row.class_id in DEBRIS_CLASS_IDS or class_name in DEBRIS_CLASS_NAMES:
                debris_names.add(class_name)

        if not debris_names:
            return

        if 'track_intrusion_debris' in debris_names:
            text = '警告，发现轨道异物'
        elif 'carriage_rim_debris' in debris_names:
            text = '警告，发现车厢沿异物'
        else:
            text = '警告，发现异物'
        self._trigger_voice_alarm('debris', text, ALARM_DEBRIS_COOLDOWN_S)

    def _check_system_alarm(self, stats: SystemStats):
        max_temp = max(stats.temp_zones.values()) if stats.temp_zones else 0
        if max_temp >= ALARM_TEMP_CRITICAL_C:
            self._trigger_voice_alarm(
                'temp_critical',
                f'严重警告，系统温度过高，当前 {max_temp:.0f} 度',
                ALARM_TEMP_COOLDOWN_S,
            )
        elif max_temp >= ALARM_TEMP_WARNING_C:
            self._trigger_voice_alarm(
                'temp_high',
                f'警告，系统温度偏高，当前 {max_temp:.0f} 度',
                ALARM_TEMP_COOLDOWN_S,
            )

        if stats.cpu_percent >= ALARM_CPU_WARNING_PERCENT:
            self._system_alarm_counts['cpu_high'] += 1
            if self._system_alarm_counts['cpu_high'] >= 3:
                self._trigger_voice_alarm(
                    'cpu_high',
                    f'警告，CPU 负载过高，当前 {stats.cpu_percent:.0f}%',
                    ALARM_RESOURCE_COOLDOWN_S,
                )
        else:
            self._system_alarm_counts['cpu_high'] = 0

        if stats.memory_percent >= ALARM_MEMORY_WARNING_PERCENT:
            self._system_alarm_counts['memory_high'] += 1
            if self._system_alarm_counts['memory_high'] >= 3:
                self._trigger_voice_alarm(
                    'memory_high',
                    f'警告，内存占用过高，当前 {stats.memory_percent:.0f}%',
                    ALARM_RESOURCE_COOLDOWN_S,
                )
        else:
            self._system_alarm_counts['memory_high'] = 0

    def _event_upload_enabled(self):
        checkbox = getattr(self.detection_page, 'upload_enable_check', None)
        return bool(checkbox and checkbox.isChecked())

    def _should_upload_event(self, key, cooldown_s):
        now = time.time()
        last = self._last_upload_by_key.get(key, 0.0)
        if now - last < cooldown_s:
            return False
        self._last_upload_by_key[key] = now
        return True

    def _system_snapshot(self):
        stats = self.system_monitor.latest()
        max_temp = max(stats.temp_zones.values()) if stats.temp_zones else 0
        return {
            'cpu_percent': round(stats.cpu_percent, 1),
            'memory_percent': round(stats.memory_percent, 1),
            'memory_used_mb': round(stats.memory_used_mb, 1),
            'process_rss_mb': round(stats.process_rss_mb, 1),
            'max_temp_c': round(max_temp, 1),
            'npu_load_percent': round(stats.npu_load_percent, 1) if stats.npu_load_percent >= 0 else None,
            'gpu_load_percent': round(stats.gpu_load_percent, 1) if stats.gpu_load_percent >= 0 else None,
            'timestamp': stats.timestamp,
        }

    def _thumbnail_payload(self, frame):
        if frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            max_w = THUMBNAIL_MAX_WIDTH
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return None
            return {
                'format': 'jpg',
                'encoding': 'base64',
                'width': int(w),
                'height': int(h),
                'data': base64.b64encode(buf.tobytes()).decode('ascii'),
            }
        except Exception:
            return None

    def _submit_detection_event(self, result: FrameResult, vis_frame=None):
        if not self._event_upload_enabled() or not result or not result.rows:
            return
        self._apply_upload_url()

        detections = []
        class_counts = {}
        ocr_texts = []
        debris_names = set()
        for row in result.rows:
            class_name = row.class_name or get_class_name(row.class_id, self.engine.classes)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if row.ocr_text:
                ocr_texts.append(row.ocr_text)
            if row.class_id in DEBRIS_CLASS_IDS or class_name in DEBRIS_CLASS_NAMES:
                debris_names.add(class_name)
            detections.append({
                'class_id': row.class_id,
                'class_name': class_name,
                'bbox': [row.x1, row.y1, row.x2, row.y2],
                'confidence': round(float(row.det_conf), 4),
                'ocr_text': row.ocr_text,
            })

        if debris_names:
            event_type = 'debris_alarm'
            severity = 'critical' if 'track_intrusion_debris' in debris_names else 'warning'
            upload_key = 'debris_alarm'
            cooldown_s = DETECTION_UPLOAD_COOLDOWN_S
            message = '发现轨道/车厢异物'
        elif ocr_texts:
            event_type = 'ocr_record'
            severity = 'info'
            upload_key = 'ocr_record'
            cooldown_s = OCR_UPLOAD_COOLDOWN_S
            message = '识别到车号区域'
        else:
            return

        if not self._should_upload_event(upload_key, cooldown_s):
            return

        payload = {
            'device_id': DEFAULT_DEVICE_ID,
            'event_type': event_type,
            'severity': severity,
            'message': message,
            'source': result.source,
            'frame': result.frame,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'detections': detections,
            'class_counts': class_counts,
            'ocr_texts': ocr_texts,
            'timing': {
                'yolo_ms': round(result.yolo_ms, 1),
                'ocr_ms': round(result.ocr_ms, 1),
                'total_ms': round(result.total_ms, 1),
                'fps': round(result.fps, 1),
            },
            'system': self._system_snapshot(),
            'thumbnail': self._thumbnail_payload(vis_frame),
        }
        path = self.event_uploader.enqueue(payload)
        self._log(f'[联网] 已加入上传队列: {event_type} frame={result.frame} ({path.name})')
        self._update_upload_status()

    def _submit_system_metric(self, stats):
        if not self._event_upload_enabled():
            return
        self._apply_upload_url()
        max_temp = max(stats.temp_zones.values()) if stats.temp_zones else 0
        payload = {
            'device_id': DEFAULT_DEVICE_ID,
            'event_type': 'system_metric',
            'severity': 'info',
            'message': '系统性能指标',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system': {
                'cpu_percent': round(stats.cpu_percent, 1),
                'memory_percent': round(stats.memory_percent, 1),
                'memory_used_mb': round(stats.memory_used_mb, 1),
                'process_rss_mb': round(stats.process_rss_mb, 1),
                'max_temp_c': round(max_temp, 1),
                'npu_load_percent': round(stats.npu_load_percent, 1) if stats.npu_load_percent >= 0 else None,
                'gpu_load_percent': round(stats.gpu_load_percent, 1) if stats.gpu_load_percent >= 0 else None,
                'cpu_freq_mhz': round(stats.cpu_freq_mhz, 0) if stats.cpu_freq_mhz > 0 else None,
            },
        }
        self.event_uploader.enqueue(payload)

    def _queue_alarm_event(self, event_key, text):
        if event_key == 'debris' or not self._event_upload_enabled():
            return
        self._apply_upload_url()
        severity = 'critical' if event_key == 'temp_critical' else 'warning'
        payload = {
            'device_id': DEFAULT_DEVICE_ID,
            'event_type': 'system_alarm',
            'alarm_key': event_key,
            'severity': severity,
            'message': text,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system': self._system_snapshot(),
        }
        path = self.event_uploader.enqueue(payload)
        self._log(f'[联网] 系统报警已加入上传队列: {event_key} ({path.name})')
        self._update_upload_status()

    def _start_voice_control(self):
        dp = self.detection_page
        if not dp.voice_enable_check.isChecked():
            self._log('语音控制未启用')
            return
        port = dp.voice_port_combo.currentText().strip()
        baud_text = dp.voice_baud_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, '提示', '请填写 LD3320 串口设备，例如 /dev/ttyS9')
            return
        try:
            baudrate = int(baud_text)
        except ValueError:
            QMessageBox.warning(self, '提示', f'波特率不是数字: {baud_text}')
            return
        self.voice_worker = VoiceSerialWorker(port, baudrate, DEFAULT_VOICE_CONFIG)
        self.voice_worker.command_received.connect(self._on_voice_command)
        self.voice_worker.raw_received.connect(self._on_voice_raw)
        self.voice_worker.status_message.connect(self._on_voice_status)
        self.voice_worker.error.connect(self._on_voice_error)
        self.voice_worker.start()
        dp.btn_voice_connect.setText('断开语音')
        dp.voice_status_label.setText('连接中...')
        dp.voice_status_label.setStyleSheet('color: #d29922; font-size: 12px;')
        self._log(f'正在连接 LD3320: {port} @ {baudrate}')

    def _stop_voice_control(self):
        if self.voice_worker and self.voice_worker.isRunning():
            self.voice_worker.request_stop()
            self.voice_worker.wait(2000)
        self.voice_worker = None
        dp = self.detection_page
        dp.btn_voice_connect.setText('连接语音')
        dp.voice_status_label.setText('未连接')
        dp.voice_status_label.setStyleSheet(STYLE_SUBTLE)

    def _on_voice_status(self, msg):
        self._log(f'[语音] {msg}')
        dp = self.detection_page
        if '已连接' in msg:
            dp.voice_status_label.setText('已连接')
            dp.voice_status_label.setStyleSheet(STYLE_SUCCESS)
        elif '断开' in msg:
            dp.voice_status_label.setText('未连接')
            dp.voice_status_label.setStyleSheet(STYLE_SUBTLE)
            dp.btn_voice_connect.setText('连接语音')

    def _on_voice_raw(self, msg):
        self._log(f'[语音] {msg}')
        self.detection_page.voice_last_label.setText('命令: 未匹配')

    def _on_voice_error(self, msg):
        self._log(f'[语音 ERROR] {msg}')
        self.statusBar().showMessage(f'语音模块错误: {msg}')
        self.detection_page.voice_status_label.setText('错误')
        self.detection_page.voice_status_label.setStyleSheet(STYLE_ERROR)
        self.detection_page.btn_voice_connect.setText('连接语音')

    def _on_voice_command(self, action, detail):
        now = time.time()
        if action == self._last_voice_action and now - self._last_voice_action_ts < 1.0:
            return
        self._last_voice_action = action
        self._last_voice_action_ts = now
        label = detail.split('|', 1)[0].strip() if detail else action
        if not self.detection_page.btn_voice_execute.isChecked():
            self._log(f'[语音识别] {detail}（执行开关关闭，未操作）')
            self.detection_page.voice_last_label.setText(f'识别: {label}（未执行）')
            self.statusBar().showMessage('语音识别已收到，执行开关关闭')
            return
        self._log(f'[语音命令] {detail}')
        self.detection_page.voice_last_label.setText(f'命令: {label}')
        self._execute_voice_action(action)

    def _execute_voice_action(self, action):
        self.tab_widget.setCurrentWidget(self.detection_page)
        if action == 'open_camera':
            self._open_camera()
            self.statusBar().showMessage('语音命令: 打开摄像头')
            return {"ok": True, "message": "已打开摄像头"}
        elif action == 'stop_detection':
            self._stop()
            self.statusBar().showMessage('语音命令: 停止检测')
            return {"ok": True, "message": "已停止检测"}
        elif action == 'pause_detection':
            if self.worker and self.worker.isRunning() and not self.worker.paused:
                self._pause()
                self.statusBar().showMessage('语音命令: 暂停检测')
                return {"ok": True, "message": "已暂停检测"}
            else:
                self._log('[语音] 当前没有正在运行的检测可暂停')
                return {"ok": False, "message": "没有正在运行的检测可暂停"}
        elif action in ('start_detection', 'resume_detection'):
            if self.worker and self.worker.isRunning() and self.worker.paused:
                self._start()
                self.statusBar().showMessage('语音命令: 继续检测')
                return {"ok": True, "message": "已继续检测"}
            elif hasattr(self, '_pending_source') and hasattr(self, '_pending_mode'):
                self._start()
                self.statusBar().showMessage('语音命令: 开始检测')
                return {"ok": True, "message": "已开始检测"}
            elif self.worker and self.worker.isRunning():
                self._log('[语音] 检测已经在运行中')
                return {"ok": False, "message": "检测已在运行中"}
            else:
                self._log('[语音] 没有待开始的视频；实时检测请说"打开摄像头"')
                return {"ok": False, "message": "没有待开始的视频"}
        elif action == 'start_capacity':
            self.tab_widget.setCurrentWidget(self.capacity_page)
            self._start_capacity_test()
            return {"ok": True, "message": "已开始评估"}
        elif action == 'stop_capacity':
            self.tab_widget.setCurrentWidget(self.capacity_page)
            self._stop_capacity_test()
            return {"ok": True, "message": "已停止评估"}
        elif action == 'toggle_recording':
            if self.detection_page.btn_record.isEnabled():
                self._toggle_recording()
                return {"ok": True, "message": "已切换录制状态"}
            else:
                self._log('[语音] 当前不能录制，请先开始视频或摄像头检测')
                return {"ok": False, "message": "当前不能录制，请先开始检测"}
        elif action == 'stop_recording':
            if self._recording:
                self._toggle_recording()
                self.statusBar().showMessage('远程命令: 停止录制')
                return {"ok": True, "message": "已停止录制"}
            else:
                self._log('[远程] 当前没有在录制')
                return {"ok": False, "message": "当前没有在录制"}
        elif action == 'show_status':
            s = self.system_monitor.latest()
            max_temp = max(s.temp_zones.values()) if s.temp_zones else 0
            msg = f'CPU {s.cpu_percent:.0f}%, 内存 {s.memory_percent:.0f}%, 温度 {max_temp:.1f}C'
            self._log(f'[语音] 系统状态: {msg}')
            return {"ok": True, "message": msg}
        elif action == 'mute_alarm':
            self.detection_page.alarm_enable_check.setChecked(False)
            self.statusBar().showMessage('语音命令: 静音报警')
            return {"ok": True, "message": "已静音报警"}
        elif action == 'unmute_alarm':
            self.detection_page.alarm_enable_check.setChecked(True)
            self.statusBar().showMessage('语音命令: 解除静音')
            return {"ok": True, "message": "已解除静音"}
        else:
            self._log(f'[语音] 未实现的动作: {action}')
            return {"ok": False, "message": f"未实现的动作: {action}"}

    def _auto_load_model(self):
        yolo_path = None
        ocr_path = None
        for p in DEFAULT_YOLO_MODELS:
            if p.exists():
                yolo_path = str(p)
                break
        for p in DEFAULT_OCR_MODELS:
            if p.exists():
                ocr_path = str(p)
                break
        if not yolo_path:
            self.statusBar().showMessage('无可用模型')
            return
        self.statusBar().showMessage(f'正在加载模型: {Path(yolo_path).name}...')
        config = ModelConfig(
            yolo_model=Path(yolo_path),
            ocr_model=Path(ocr_path) if ocr_path else Path(''),
            classes_file=DEFAULT_CLASSES_FILE,
            dict_file=DEFAULT_DICT_FILE,
            skip_ocr=not ocr_path,
        )
        self._load_worker = ModelLoadWorker(config)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.start()

    def _on_model_loaded(self, engine, error):
        if engine:
            self.engine = engine
            self.statusBar().showMessage('模型已加载，就绪')
            self._log('模型加载成功')
        else:
            self.statusBar().showMessage(f'模型加载失败: {error}')
            self._log(f'模型加载失败: {error}')

    def _show_model_switch_dialog(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, '提示', '请先停止当前任务')
            return
        yolo_path, _ = QFileDialog.getOpenFileName(
            self, '选择 YOLO 模型', '', '模型文件 (*.pt *.onnx *.rknn)')
        if not yolo_path:
            return
        ocr_path, _ = QFileDialog.getOpenFileName(
            self, '选择 OCR 模型', '', '模型文件 (*.pth *.onnx *.rknn)')
        self.statusBar().showMessage(f'正在切换模型: {Path(yolo_path).name}...')
        if self.engine:
            self.engine.close()
        config = ModelConfig(
            yolo_model=Path(yolo_path),
            ocr_model=Path(ocr_path) if ocr_path else Path(''),
            classes_file=DEFAULT_CLASSES_FILE,
            dict_file=DEFAULT_DICT_FILE,
            skip_ocr=not ocr_path,
        )
        self._load_worker = ModelLoadWorker(config)
        self._load_worker.finished.connect(self._on_model_loaded)
        self._load_worker.start()

    # ── 输入源 ──

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '打开图片', '', '图片文件 (*.jpg *.jpeg *.png *.bmp)')
        if not path:
            return
        if not self.engine:
            QMessageBox.warning(self, '提示', '模型未加载')
            return
        self._start_inference('image', path)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '打开视频', '', '视频文件 (*.mp4 *.avi *.mkv *.mov)')
        if not path:
            return
        if not self.engine:
            QMessageBox.warning(self, '提示', '模型未加载')
            return
        self._pending_source = path
        self._pending_mode = 'video'
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimage = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
                self._current_pixmap = QPixmap.fromImage(qimage)
                label = self.detection_page.image_label
                label.setPixmap(self._current_pixmap.scaled(
                    label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            cap.release()
        dp = self.detection_page
        dp.btn_start.setEnabled(True)
        dp.btn_pause.setEnabled(False)
        dp.btn_stop.setEnabled(False)
        dp.btn_record.setEnabled(False)
        self.statusBar().showMessage(f'已加载视频: {Path(path).name}，点击"开始"播放')

    def _open_camera(self):
        if not self.engine:
            QMessageBox.warning(self, '提示', '模型未加载')
            return
        default_device = '/dev/video11' if sys.platform == 'linux' else '0'
        self._start_inference('camera', default_device)

    # ── 播放控制 ──

    def _start(self):
        if self.worker and self.worker.isRunning():
            if self.worker.paused:
                self.worker.request_resume()
                self.detection_page.btn_start.setEnabled(False)
                self.detection_page.btn_pause.setEnabled(True)
                self.statusBar().showMessage('运行中')
                self._log('继续播放')
            return
        if hasattr(self, '_pending_source') and hasattr(self, '_pending_mode'):
            self._start_inference(self._pending_mode, self._pending_source)

    def _start_inference(self, mode, source):
        self._stop()
        self.engine.config.conf = self.detection_page.conf_spin.value()
        self.engine.config.iou = self.detection_page.iou_spin.value()
        self.engine.config.skip_ocr = False
        # 图片模式关闭时序过滤（单帧无法满足多帧确认），视频模式开启
        self.engine.set_temporal_enabled(mode != 'image')
        # 切换视频源时重置时序历史
        self.engine.reset_temporal()

        self.worker = InferenceWorker(self.engine)
        self.worker.mode = mode
        self.worker.source = source
        self.worker.every_n = self.detection_page.every_n_spin.value()
        self.worker.output_dir = self.detection_page.output_dir_edit.text()

        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.log_message.connect(self._log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)

        dp = self.detection_page
        dp.btn_start.setEnabled(False)
        dp.btn_pause.setEnabled(True)
        dp.btn_stop.setEnabled(True)
        dp.btn_record.setEnabled(True)

        self.statusBar().showMessage(f'运行中: {mode} - {source}')

        metrics_dir = Path(self.detection_page.output_dir_edit.text()) / 'logs'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.system_monitor.start(csv_path=metrics_dir / f'metrics_{time.strftime("%Y%m%d_%H%M%S")}.csv')

        self.worker.start()
        self._log(f'开始推理: {mode}')

    def _pause(self):
        if self.worker and self.worker.isRunning() and not self.worker.paused:
            self.worker.request_pause()
            self.detection_page.btn_start.setEnabled(True)
            self.statusBar().showMessage('已暂停')

    def _reset_recording_ui(self):
        dp = self.detection_page
        self._recording = False
        dp.record_indicator.setText('⚫ 未录制')
        dp.btn_record.setText('开始录制')
        dp.btn_record.setObjectName('btn_record')
        dp.btn_record.style().unpolish(dp.btn_record)
        dp.btn_record.style().polish(dp.btn_record)

    def _stop(self):
        dp = self.detection_page
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(WORKER_WAIT_TIMEOUT_MS)
        dp.btn_start.setEnabled(False)
        dp.btn_pause.setEnabled(False)
        dp.btn_stop.setEnabled(False)
        dp.btn_record.setEnabled(False)
        self._reset_recording_ui()

    def _toggle_recording(self):
        dp = self.detection_page
        if not self._recording:
            self._recording = True
            dp.btn_record.setText('停止录制')
            dp.btn_record.setObjectName('btn_record_active')
            dp.record_indicator.setText('🔴 录制中')
            if self.worker and self.worker.isRunning():
                self.worker.request_start_recording()
            self._log('开始录制')
        else:
            self._recording = False
            dp.btn_record.setText('开始录制')
            dp.btn_record.setObjectName('btn_record')
            dp.record_indicator.setText('⚫ 未录制')
            if self.worker and self.worker.isRunning():
                self.worker.request_stop_recording()
            self._log('停止录制')
        dp.btn_record.style().unpolish(dp.btn_record)
        dp.btn_record.style().polish(dp.btn_record)

    # ── 导出 ──

    def _export_results(self):
        if not self._last_session_dir:
            QMessageBox.information(self, '提示', '暂无结果可导出，请先运行一次推理')
            return
        session_path = Path(self._last_session_dir)
        if not session_path.exists():
            QMessageBox.warning(self, '提示', f'结果目录不存在:\n{session_path}\n\n请先运行一次推理')
            return
        dp = self.detection_page
        want_csv = dp.export_csv_check.isChecked()
        want_img = dp.export_img_check.isChecked()
        want_video = dp.export_video_check.isChecked()
        if not want_csv and not want_img and not want_video:
            QMessageBox.information(self, '提示', '请至少选择一种导出格式')
            return
        files = []
        for f in session_path.iterdir():
            ext = f.suffix.lower()
            if want_csv and ext == '.csv':
                files.append(f)
            elif want_img and ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                files.append(f)
            elif want_video and ext in ('.mp4', '.avi', '.mkv'):
                files.append(f)
        if not files:
            QMessageBox.information(self, '提示', '所选格式下没有可导出的文件')
            return
        export_dir = QFileDialog.getExistingDirectory(self, '选择导出目标目录')
        if not export_dir:
            return
        import shutil
        try:
            target = Path(export_dir) / session_path.name
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(str(f), str(target / f.name))
            self._log(f'已导出 {len(files)} 个文件到: {target}')
            QMessageBox.information(self, '导出成功', f'已导出 {len(files)} 个文件到:\n{target}')
        except Exception as e:
            self._log(f'导出失败: {e}')
            QMessageBox.warning(self, '导出失败', str(e))

    # ── 扩展能力测试 ──

    def _start_capacity_test(self):
        if not self.engine:
            QMessageBox.warning(self, '提示', '模型未加载')
            return
        video_source = self.capacity_page.video_source_edit.text()
        if video_source == '未选择' or not Path(video_source).exists():
            QMessageBox.warning(self, '提示', '请先选择有效的测试视频')
            return

        cp = self.capacity_page
        cp.btn_start_capacity.setEnabled(False)
        cp.btn_stop_capacity.setEnabled(True)
        cp.capacity_status.setText('运行中...')
        cp.capacity_status.setStyleSheet('color: #3fb950;')
        cp.capacity_progress.setText('准备开始...')
        cp.capacity_chart.clear_data()
        cp.result_table.setRowCount(0)

        output_dir = Path(self.detection_page.output_dir_edit.text()) / 'capacity'

        from src.capacity.tbju_capacity_test import DEFAULT_PROFILES
        self.capacity_worker = CapacityWorker(
            self.engine, video_source, DEFAULT_PROFILES, output_dir,
            self.system_monitor
        )
        self.capacity_worker.progress.connect(self._on_capacity_progress)
        self.capacity_worker.metrics.connect(self._on_capacity_metrics)
        self.capacity_worker.finished.connect(self._on_capacity_finished)
        self.capacity_worker.error.connect(self._on_capacity_error)
        self.capacity_worker.start()

        self._log('扩展能力评估已开始')

    def _stop_capacity_test(self):
        if self.capacity_worker:
            self.capacity_worker.request_stop()
            self._log('正在停止扩展能力评估...')

    def _on_capacity_progress(self, msg):
        self.capacity_page.capacity_progress.setText(msg)

    def _on_capacity_metrics(self, data):
        cp = self.capacity_page
        cp.capacity_chart.add_point('total_ms', data.get('total_ms', 0))
        cp.capacity_chart.add_point('yolo_ms', data.get('yolo_ms', 0))

    def _on_capacity_finished(self, session_dir):
        cp = self.capacity_page
        cp.btn_start_capacity.setEnabled(True)
        cp.btn_stop_capacity.setEnabled(False)
        cp.capacity_status.setText('完成')
        cp.capacity_status.setStyleSheet('color: #58a6ff;')

        if session_dir:
            self._last_report_dir = session_dir
            cp.btn_open_report_dir.setEnabled(True)
            self._load_capacity_results(session_dir)
            self._log(f'扩展能力评估完成，报告: {session_dir}')
        else:
            self._log('扩展能力评估完成（无结果）')

    def _on_capacity_error(self, msg):
        self._log(f'[ERROR] 压测错误: {msg}')
        self.capacity_page.capacity_status.setText(f'错误: {msg}')
        self.capacity_page.capacity_status.setStyleSheet('color: #f85149;')

    def _load_capacity_results(self, session_dir):
        import json
        json_path = Path(session_dir) / 'capacity_report.json'
        if not json_path.exists():
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cp = self.capacity_page
        cp.result_table.setRowCount(len(data))
        for i, r in enumerate(data):
            cp.result_table.setItem(i, 0, QTableWidgetItem(r.get('profile_name', '')))
            cp.result_table.setItem(i, 1, QTableWidgetItem(str(r.get('roi_count', ''))))
            cp.result_table.setItem(i, 2, QTableWidgetItem(str(r.get('every_n', ''))))
            cp.result_table.setItem(i, 3, QTableWidgetItem(f"{r.get('avg_infer_fps', 0):.1f}"))
            cp.result_table.setItem(i, 4, QTableWidgetItem(f"{r.get('p50_total_ms', 0):.0f}"))
            cp.result_table.setItem(i, 5, QTableWidgetItem(f"{r.get('p95_total_ms', 0):.0f}"))
            cp.result_table.setItem(i, 6, QTableWidgetItem(f"{r.get('avg_cpu_percent', 0):.0f}"))
            cp.result_table.setItem(i, 7, QTableWidgetItem(f"{r.get('max_temp_c', 0):.1f}"))
            cp.result_table.setItem(i, 8, QTableWidgetItem(f"{r.get('avg_npu_load_percent', 0):.0f}"))
            cp.result_table.setItem(i, 9, QTableWidgetItem(r.get('recommendation', '')))

    def _open_report_dir(self):
        if self._last_report_dir:
            import subprocess
            path = str(self._last_report_dir)
            if sys.platform == 'win32':
                subprocess.Popen(['explorer', path.replace('/', '\\')])
            elif sys.platform == 'linux':
                subprocess.Popen(['xdg-open', path])

    # ── 回调 ──

    def _on_frame_ready(self, vis_frame, result):
        if vis_frame is not None:
            rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimage = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)
            self._current_pixmap = QPixmap.fromImage(qimage)
            label = self.detection_page.image_label
            label.setPixmap(self._current_pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if result is not None:
            self.current_result = result
            self._update_stats(result)
            self._check_detection_alarm(result)
            self._submit_detection_event(result, vis_frame)
            self.detection_page.btn_export.setEnabled(True)

    def _on_error(self, msg):
        self._log(f'[ERROR] {msg}')
        self.statusBar().showMessage(f'错误: {msg}')

    def _on_finished(self, session_dir):
        self._last_session_dir = session_dir
        dp = self.detection_page
        dp.btn_start.setEnabled(False)
        dp.btn_pause.setEnabled(False)
        dp.btn_stop.setEnabled(False)
        dp.btn_record.setEnabled(False)
        self._reset_recording_ui()
        self.system_monitor.stop()
        self.statusBar().showMessage('就绪')
        # 同步 CSV 文件到看板（延时确保文件已完全关闭）
        if session_dir and not self._closing:
            import threading as _threading
            self._sync_timer = _threading.Timer(1.0, self._sync_csv_to_dashboard, args=(session_dir,))
            self._sync_timer.daemon = True
            self._sync_timer.start()

    def _sync_csv_to_dashboard(self, session_dir: str):
        """将 session 目录和 logs 目录中的 CSV 文件同步到看板（去重 + 重试）。"""
        if self._closing:
            return
        if not self.event_uploader or not self.event_uploader.server_url:
            return
        session_path = Path(session_dir)
        if not session_path.exists():
            return
        session_name = session_path.name
        # 扫描 session 目录（result.csv）+ logs 目录（metrics.csv）
        csv_files = list(session_path.glob("*.csv"))
        logs_dir = DEFAULT_OUTPUT_DIR / 'logs'
        if logs_dir.exists():
            csv_files.extend(logs_dir.glob("metrics_*.csv"))
        if not csv_files:
            return

        # 去重：线程安全的 set + lock + 持久化到磁盘
        if not hasattr(self, '_synced_csv_files'):
            self._synced_csv_lock = threading.Lock()
            synced_file = DEFAULT_OUTPUT_DIR / '.csv_synced'
            self._synced_csv_path = synced_file
            if synced_file.exists():
                self._synced_csv_files = set(synced_file.read_text(encoding='utf-8').splitlines())
            else:
                self._synced_csv_files = set()

        with self._synced_csv_lock:
            new_files = []
            for f in csv_files:
                key = f'{session_name}/{f.name}'
                if key not in self._synced_csv_files:
                    new_files.append(f)
                    self._synced_csv_files.add(key)  # 预占，防止并发重复
        if not new_files:
            return

        def _upload():
            uploaded = 0
            for csv_file in new_files:
                file_type = "metrics" if "metrics" in csv_file.name else "result"
                # 来源目录：session 的父目录名（camera/images/videos）或 logs
                source_dir = csv_file.parent.name if csv_file.parent.name != session_name else session_path.parent.name
                key = f'{session_name}/{csv_file.name}'
                success = False
                for attempt in range(3):
                    if self.event_uploader.upload_csv_file(csv_file, session_name, file_type, source_dir):
                        success = True
                        break
                    time.sleep(1)
                if success:
                    uploaded += 1
                else:
                    # 上传失败，移除预占标记，允许下次重试
                    with self._synced_csv_lock:
                        self._synced_csv_files.discard(key)
                    self._log(f'[同步] 上传失败: {csv_file.name}（已重试3次）')
            if uploaded > 0:
                # 持久化已上传列表到磁盘
                with self._synced_csv_lock:
                    try:
                        self._synced_csv_path.write_text(
                            '\n'.join(sorted(self._synced_csv_files)), encoding='utf-8')
                    except Exception:
                        pass
                self._log(f'[同步] 已上传 {uploaded} 个 CSV 文件到看板')

        threading.Thread(target=_upload, daemon=True, name='CSVSync').start()

    def _update_stats(self, result: FrameResult):
        counts = {}
        ocr_results = []
        for row in result.rows:
            name = get_class_name(row.class_id, self.engine.classes)
            counts[name] = counts.get(name, 0) + 1
            if row.ocr_text:
                ocr_results.append(row.ocr_text)
        lines = [f'检测: {len(result.rows)} 个目标']
        for name, count in counts.items():
            lines.append(f'  {name}: {count}')
        if ocr_results:
            lines.append(f'\n车号: {", ".join(ocr_results)}')
        lines.append(f'\nYOLO: {result.yolo_ms:.1f}ms')
        lines.append(f'OCR: {result.ocr_ms:.1f}ms')
        if result.fps > 0:
            lines.append(f'FPS: {result.fps:.1f}')
        self.detection_page.stats_label.setText('\n'.join(lines))

    def _update_monitor(self):
        stats = self.system_monitor.latest()
        # Tab1: 简洁监控
        self.detection_page.monitor_label.setText(self.system_monitor.format_display())
        # Tab2: 详细监控 + 曲线
        self.perf_page.perf_monitor_label.setText(self.system_monitor.format_display(show_sources=True))
        # GPU 状态
        gpu_load = stats.gpu_load_percent
        gpu_lines = []
        if gpu_load >= 0:
            gpu_lines.append(f'GPU 负载: {gpu_load:.0f}%')
            if stats.gpu_freq_mhz > 0:
                gpu_lines.append(f'GPU 频率: {stats.gpu_freq_mhz:.0f} MHz')
        else:
            gpu_lines.append('GPU: 未检测到')
        self.perf_page.npu_label.setText('\n'.join(gpu_lines))
        # 曲线
        self.perf_page.chart_cpu.add_point('CPU', stats.cpu_percent)
        self.perf_page.chart_cpu.add_point('内存', stats.memory_percent)
        max_temp = max(stats.temp_zones.values()) if stats.temp_zones else 0
        self.perf_page.chart_temp.add_point('温度', max_temp)
        self._check_system_alarm(stats)
        # 历史表格
        self._monitor_history.append(stats)
        self._update_history_table()
        # 带宽监控
        bw = self.event_uploader.get_bandwidth()
        bw_kbps = bw['bytes_per_sec'] / 1024
        self._peak_bw_kbps = max(getattr(self, '_peak_bw_kbps', 0), bw_kbps)
        self.perf_page.chart_bandwidth.add_point('上传', bw_kbps)
        # 动态调整 Y 轴
        if bw_kbps > self.perf_page.chart_bandwidth.y_range[1] * 0.8:
            new_max = max(100, bw_kbps * 2)
            self.perf_page.chart_bandwidth.set_y_range(0, new_max)
        self.perf_page.bandwidth_label.setText(
            f'当前: {bw_kbps:.1f} KB/s\n'
            f'峰值: {self._peak_bw_kbps:.1f} KB/s\n'
            f'累计: {bw["total_kb"]:.1f} KB'
        )
        # 每 5 秒上传一次系统性能指标到远程看板
        self._metric_upload_tick += 1
        if self._metric_upload_tick >= 5:
            self._metric_upload_tick = 0
            self._submit_system_metric(stats)

    def _update_history_table(self):
        table = self.perf_page.history_table
        rows = list(self._monitor_history)
        if table.rowCount() != len(rows):
            table.setRowCount(len(rows))
        for i, s in enumerate(reversed(rows)):
            vals = [
                s.timestamp.split()[-1] if s.timestamp else '',
                f'{s.cpu_percent:.0f}',
                f'{s.memory_used_mb:.0f}',
                f'{max(s.temp_zones.values()):.1f}' if s.temp_zones else '0.0',
                f'{s.cpu_freq_mhz:.0f}' if s.cpu_freq_mhz > 0 else 'N/A',
            ]
            for j, val in enumerate(vals):
                item = table.item(i, j)
                if item is None:
                    table.setItem(i, j, QTableWidgetItem(val))
                else:
                    item.setText(val)

    def _browse_output_dir(self):
        current = self.detection_page.output_dir_edit.text()
        d = QFileDialog.getExistingDirectory(self, '选择输出目录', current)
        if d:
            self.detection_page.output_dir_edit.setText(d)
            self._log(f'输出目录已更新: {d}')

    @staticmethod
    def _sanitize_log(msg: str) -> str:
        """脱敏日志内容：移除文件路径和 IP 地址。"""
        clean = re.sub(r'[A-Za-z]:\\[^\s]+', '<path>', msg)
        clean = re.sub(r'/[\w./\-]+', '<path>', clean)
        clean = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<ip>', clean)
        return clean

    def _log(self, msg):
        self.detection_page.log_text.append(
            f'<span style="color:#484f58">[{time.strftime("%H:%M:%S")}]</span> {msg}')
        if self.event_uploader and self.event_uploader.server_url:
            try:
                self._log_counter += 1
                is_warning = any(kw in msg for kw in ('ERROR', 'WARN', '警告', '严重', '失败'))
                if is_warning or self._log_counter % LOG_UPLOAD_SAMPLE_INTERVAL == 0:
                    clean_msg = self._sanitize_log(re.sub(r'<[^>]+>', '', msg))
                    self.event_uploader.enqueue({
                        'event_type': 'app_log',
                        'device_id': DEFAULT_DEVICE_ID,
                        'message': clean_msg,
                        'created_at': datetime.now().isoformat(),
                    })
            except Exception:
                pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap:
            label = self.detection_page.image_label
            label.setPixmap(self._current_pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self._closing = True
        if self._sync_timer and self._sync_timer.is_alive():
            self._sync_timer.cancel()
        self._stop_voice_control()
        self.voice_alarm.close()
        self.event_uploader.close()
        if self.command_poller:
            self.command_poller.close()
        self._stop()
        if self.capacity_worker and self.capacity_worker.isRunning():
            self.capacity_worker.request_stop()
            self.capacity_worker.wait(3000)
        # 等待模型加载线程结束
        if hasattr(self, '_load_worker') and self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait(3000)
        if self.engine:
            self.engine.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_STYLE)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor('#0d1117'))
    palette.setColor(QPalette.WindowText, QColor('#c9d1d9'))
    palette.setColor(QPalette.Base, QColor('#161b22'))
    palette.setColor(QPalette.AlternateBase, QColor('#0d1117'))
    palette.setColor(QPalette.Text, QColor('#c9d1d9'))
    palette.setColor(QPalette.Button, QColor('#21262d'))
    palette.setColor(QPalette.ButtonText, QColor('#c9d1d9'))
    palette.setColor(QPalette.Highlight, QColor('#1a2332'))
    palette.setColor(QPalette.HighlightedText, QColor('#f0f6fc'))
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    # 强制居中并最大化（兼容 RK3588 桌面环境）
    screen = app.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        window.setGeometry(geo)
    else:
        window.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
