#!/usr/bin/env python3
"""
tbju_demo_tk.py — 列车车厢检测识别系统 GUI (tkinter 版)
当 PyQt5 不可用时使用此版本。
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.tbju_rknn_core import (
    ModelConfig, TBJURKNNEngine, ResultWriter, FrameResult,
    get_class_name,
)
from src.monitor.tbju_system_monitor import SystemMonitor

DEPLOY_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = DEPLOY_DIR.parent
CARTRIDGE_ROOT = PROJECT_ROOT.parent / 'Carriage'

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


class InferenceWorker(threading.Thread):
    def __init__(self, engine, callback, log_callback, error_callback, finish_callback):
        super().__init__(daemon=True)
        self.engine = engine
        self.callback = callback
        self.log_callback = log_callback
        self.error_callback = error_callback
        self.finish_callback = finish_callback
        self.stop_requested = False
        self.paused = False
        self.mode = None
        self.source = None
        self.every_n = 2
        self.output_dir = None
        self._writer = None
        self._record_start_requested = False
        self._record_stop_requested = False
        self._is_recording = False

    def request_stop(self):
        self.stop_requested = True

    def request_pause(self):
        self.paused = True

    def request_resume(self):
        self.paused = False

    def request_start_recording(self):
        self._record_start_requested = True

    def request_stop_recording(self):
        self._record_stop_requested = True

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
            self.error_callback(str(e))
        finally:
            self.finish_callback(session_dir)

    def _run_image(self):
        image_path = Path(self.source)
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.error_callback(f'图片读取失败: {image_path}')
            return ''
        session_dir = Path(self.output_dir) / 'images' / time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(parents=True, exist_ok=True)
        writer = ResultWriter(session_dir)
        writer.open_csv()
        result = self.engine.infer_frame(frame, image_path.name, 0)
        vis = self.engine.draw(frame, result)
        writer.write_rows(result.rows, result.yolo_ms, result.ocr_ms, result.total_ms)
        writer.save_image(vis, f'{image_path.stem}_result.jpg')
        writer.close()
        self.callback(vis, result)
        self.log_callback(f'检测完成: {len(result.rows)} 个目标, YOLO {result.yolo_ms:.1f}ms, OCR {result.ocr_ms:.1f}ms')
        return str(session_dir)

    def _run_video(self):
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            self.error_callback(f'视频打开失败: {self.source}')
            return ''
        ret, frame = cap.read()
        if not ret:
            self.error_callback('视频读取首帧失败')
            cap.release()
            return ''
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or fps > 120:
            fps = 25.0
        session_dir = Path(self.output_dir) / 'videos' / time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(parents=True, exist_ok=True)
        writer = ResultWriter(session_dir)
        writer.open_csv()
        self._writer = writer
        source_name = Path(self.source).name
        frame_id = 0
        processed = 0
        last_result = None
        start = time.time()
        try:
            while not self.stop_requested:
                if self.paused:
                    time.sleep(0.05)
                    continue
                if frame_id % self.every_n == 0:
                    result = self.engine.infer_frame(frame, source_name, frame_id)
                    last_result = result
                    writer.write_rows(result.rows, result.yolo_ms, result.ocr_ms, result.total_ms)
                    processed += 1
                    elapsed = max(time.time() - start, 1e-6)
                    result.fps = processed / elapsed
                    vis = self.engine.draw(frame, result, f'FPS {result.fps:.1f} | YOLO {result.yolo_ms:.1f}ms | OCR {result.ocr_ms:.1f}ms | frame {frame_id}')
                    self.callback(vis, result)
                else:
                    vis = self.engine.draw(frame, last_result, f'skip | every_n={self.every_n} | frame {frame_id}') if last_result else frame.copy()
                    self.callback(vis, None)
                self._handle_recording(writer, session_dir, fps, width, height, vis)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
        finally:
            self._finalize_recording(writer)
            cap.release()
            writer.close()
            self._writer = None
            self.log_callback(f'视频处理完成, 保存到 {session_dir}')
        return str(session_dir)

    def _run_camera(self):
        source = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.error_callback(f'摄像头打开失败: {self.source}')
            return ''
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        if not ret:
            self.error_callback('摄像头读取首帧失败')
            cap.release()
            return ''
        session_dir = Path(self.output_dir) / 'camera' / time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(parents=True, exist_ok=True)
        writer = ResultWriter(session_dir)
        writer.open_csv()
        self._writer = writer
        frame_id = 0
        processed = 0
        last_result = None
        start = time.time()
        try:
            while not self.stop_requested:
                if self.paused:
                    time.sleep(0.05)
                    continue
                if frame_id % self.every_n == 0:
                    result = self.engine.infer_frame(frame, 'camera', frame_id)
                    last_result = result
                    writer.write_rows(result.rows, result.yolo_ms, result.ocr_ms, result.total_ms)
                    processed += 1
                    elapsed = max(time.time() - start, 1e-6)
                    result.fps = processed / elapsed
                    vis = self.engine.draw(frame, result, f'FPS {result.fps:.1f} | YOLO {result.yolo_ms:.1f}ms | OCR {result.ocr_ms:.1f}ms | frame {frame_id}')
                    self.callback(vis, result)
                else:
                    vis = self.engine.draw(frame, last_result, f'skip | every_n={self.every_n} | frame {frame_id}') if last_result else frame.copy()
                    self.callback(vis, None)
                self._handle_recording(writer, session_dir, 30.0, 640, 480, vis)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
        finally:
            self._finalize_recording(writer)
            cap.release()
            writer.close()
            self._writer = None
            self.log_callback(f'摄像头停止, 保存到 {session_dir}')
        return str(session_dir)

    def _handle_recording(self, writer, session_dir, fps, width, height, vis):
        if self._record_start_requested:
            self._record_start_requested = False
            if not self._is_recording:
                try:
                    writer.open_video(session_dir / 'record.mp4', fps, width, height)
                    self._is_recording = True
                    self.log_callback('录制开始')
                except Exception as e:
                    self.log_callback(f'录制启动失败: {e}')
        if self._record_stop_requested:
            self._record_stop_requested = False
            if self._is_recording and writer.video_writer:
                writer.video_writer.release()
                writer.video_writer = None
                self._is_recording = False
                self.log_callback(f'录制已保存: {session_dir / "record.mp4"}')
        if self._is_recording and writer.video_writer:
            writer.write_frame(vis)

    def _finalize_recording(self, writer):
        if self._is_recording and writer.video_writer:
            writer.video_writer.release()
            writer.video_writer = None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('TBJU 列车车厢检测识别系统')
        self.root.geometry('1600x900')
        self.root.configure(bg='#0d1117')
        self.engine = None
        self.worker = None
        self.system_monitor = SystemMonitor()
        self._last_session_dir = None
        self._recording = False
        self._last_frame = None
        self._active_scroll_canvas = None
        self._init_ui()
        self._auto_load_model()
        self._update_monitor()

    def _make_scrollable_frame(self, parent, width=None, fit_width=False):
        """创建可滚动的 Frame，支持垂直滚动；fit_width=True 时内部宽度锁定为面板宽度"""
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, bg='#0d1117', highlightthickness=0, width=width if width else 0)
        v_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=v_scroll.set)
        # 锁定内部宽度 = canvas 宽度，消除横向滚动
        if fit_width:
            def _sync_width(event):
                canvas.itemconfig(win_id, width=event.width)
            canvas.bind('<Configure>', _sync_width)
        # 记录鼠标进入哪个 canvas，用于全局滚轮分发
        canvas.bind('<Enter>', lambda e, c=canvas: self._on_scroll_enter(c))
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return container, inner

    def _on_scroll_enter(self, canvas):
        self._active_scroll_canvas = canvas

    def _find_scroll_target(self, event):
        """找到鼠标所在的可滚动控件：优先 Text，其次 canvas"""
        try:
            w = event.widget.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return None
        while w:
            if isinstance(w, tk.Text):
                return ('text', w)
            if isinstance(w, tk.Canvas) and w == self._active_scroll_canvas:
                return ('canvas', w)
            w = w.master
        return ('canvas', self._active_scroll_canvas) if self._active_scroll_canvas else None

    def _bind_global_scroll(self):
        """绑定全局滚轮事件，鼠标在 Text 上滚 Text，否则滚 canvas"""
        def _on_mousewheel(event):
            target = self._find_scroll_target(event)
            if not target:
                return
            kind, w = target
            if kind == 'text':
                w.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            else:
                w.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        def _on_linux_scroll_up(event):
            target = self._find_scroll_target(event)
            if not target:
                return
            kind, w = target
            w.yview_scroll(-1, 'units')
        def _on_linux_scroll_down(event):
            target = self._find_scroll_target(event)
            if not target:
                return
            kind, w = target
            w.yview_scroll(1, 'units')
        self.root.bind_all('<MouseWheel>', _on_mousewheel)
        self.root.bind_all('<Button-4>', _on_linux_scroll_up)
        self.root.bind_all('<Button-5>', _on_linux_scroll_down)

    def _init_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0d1117')
        style.configure('TLabel', background='#0d1117', foreground='#c9d1d9', font=('Microsoft YaHei UI', 9))
        style.configure('TButton', background='#21262d', foreground='#c9d1d9', font=('Microsoft YaHei UI', 8, 'bold'), padding=0)
        style.configure('TCheckbutton', background='#0d1117', foreground='#c9d1d9', font=('Microsoft YaHei UI', 8))
        style.configure('TLabelframe', background='#161b22', foreground='#58a6ff', font=('Microsoft YaHei UI', 8, 'bold'))
        style.configure('TLabelframe.Label', background='#161b22', foreground='#58a6ff', font=('Microsoft YaHei UI', 8, 'bold'))

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=0)

        # 左侧（可滚动）
        left_outer, left_frame = self._make_scrollable_frame(main_frame, width=240, fit_width=True)
        left_outer.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 2))

        input_frame = ttk.LabelFrame(left_frame, text='输入源')
        input_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        ttk.Button(input_frame, text='图片', command=self._open_image).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(input_frame, text='视频', command=self._open_video).pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(input_frame, text='摄像头', command=self._open_camera).pack(fill=tk.X, padx=2, pady=1)

        ctrl_frame = ttk.LabelFrame(left_frame, text='播放控制')
        ctrl_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        btn_row = ttk.Frame(ctrl_frame)
        btn_row.pack(fill=tk.X, padx=2, pady=1)
        ttk.Button(btn_row, text='开始', command=self._start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row, text='暂停', command=self._pause).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row, text='停止', command=self._stop).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        rec_frame = ttk.LabelFrame(left_frame, text='录制控制')
        rec_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        rec_row = ttk.Frame(rec_frame)
        rec_row.pack(fill=tk.X, padx=2, pady=1)
        self.btn_record = ttk.Button(rec_row, text='录制', command=self._toggle_recording)
        self.btn_record.pack(side=tk.LEFT, padx=2)
        self.record_label = ttk.Label(rec_row, text='⚫ 未录制', foreground='#f85149')
        self.record_label.pack(side=tk.LEFT, padx=4)

        param_frame = ttk.LabelFrame(left_frame, text='检测参数')
        param_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        ttk.Label(param_frame, text='置信度:').grid(row=0, column=0, padx=2, pady=1, sticky='w')
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(param_frame, from_=0.05, to=0.95, increment=0.05, textvariable=self.conf_var, width=6).grid(row=0, column=1, padx=2, pady=1)
        ttk.Label(param_frame, text='IoU:').grid(row=1, column=0, padx=2, pady=1, sticky='w')
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Spinbox(param_frame, from_=0.1, to=0.9, increment=0.05, textvariable=self.iou_var, width=6).grid(row=1, column=1, padx=2, pady=1)
        ttk.Label(param_frame, text='帧间隔:').grid(row=2, column=0, padx=2, pady=1, sticky='w')
        self.every_n_var = tk.IntVar(value=2)
        ttk.Spinbox(param_frame, from_=1, to=30, textvariable=self.every_n_var, width=6).grid(row=2, column=1, padx=2, pady=1)
        self.skip_ocr_var = tk.BooleanVar()
        ttk.Checkbutton(param_frame, text='跳过OCR', variable=self.skip_ocr_var).grid(row=3, column=0, columnspan=2, padx=2, pady=1, sticky='w')

        export_frame = ttk.LabelFrame(left_frame, text='导出')
        export_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        dir_row = ttk.Frame(export_frame)
        dir_row.pack(fill=tk.X, padx=2, pady=1)
        self.output_dir_var = tk.StringVar(value=str(DEPLOY_DIR / 'output'))
        ttk.Label(dir_row, textvariable=self.output_dir_var, wraplength=120).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_row, text='浏览', command=self._browse_output).pack(side=tk.RIGHT, padx=2)
        fmt_row = ttk.Frame(export_frame)
        fmt_row.pack(fill=tk.X, padx=2, pady=1)
        self.export_csv_var = tk.BooleanVar(value=True)
        self.export_img_var = tk.BooleanVar(value=True)
        self.export_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(fmt_row, text='CSV', variable=self.export_csv_var).pack(side=tk.LEFT, padx=1)
        ttk.Checkbutton(fmt_row, text='图片', variable=self.export_img_var).pack(side=tk.LEFT, padx=1)
        ttk.Checkbutton(fmt_row, text='视频', variable=self.export_video_var).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_frame, text='导出', command=self._export_results).pack(fill=tk.X, padx=2, pady=1)

        stats_frame = ttk.LabelFrame(left_frame, text='检测统计')
        stats_frame.pack(fill=tk.X, pady=(0, 4), padx=2)
        self.stats_text = tk.Text(stats_frame, bg='#161b22', fg='#f0f6fc', font=('Consolas', 9), height=5, relief=tk.FLAT, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, padx=2, pady=2)
        self.stats_text.insert(tk.END, '等待检测...')
        self.stats_text.config(state=tk.DISABLED)

        # 中央画面
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas = tk.Canvas(center_frame, bg='#010409', highlightthickness=2, highlightbackground='#21262d')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 右侧（可滚动）
        right_outer, right_frame = self._make_scrollable_frame(main_frame, width=260, fit_width=True)
        right_outer.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(2, 0))

        log_frame = ttk.LabelFrame(right_frame, text='日志')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4), padx=4)
        self.log_text = tk.Text(log_frame, bg='#0d1117', fg='#c9d1d9', font=('Consolas', 9), relief=tk.FLAT, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        monitor_frame = ttk.LabelFrame(right_frame, text='系统监控')
        monitor_frame.pack(fill=tk.X, pady=(0, 4), padx=4)
        self.monitor_text = tk.Text(monitor_frame, bg='#161b22', fg='#c9d1d9', font=('Consolas', 10), height=6, relief=tk.FLAT, wrap=tk.WORD)
        self.monitor_text.pack(fill=tk.X, padx=4, pady=4)
        self.monitor_text.insert(tk.END, 'CPU: --\n内存: --\n温度: --\nNPU: --\nGPU: --')
        self.monitor_text.config(state=tk.DISABLED)

        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        # 绑定全局滚轮（在左右面板都创建完之后）
        self._bind_global_scroll()

        # 探测系统能力并打印日志
        self.system_monitor.probe()
        probe_log = self.system_monitor.get_probe_log()
        if probe_log:
            self._log(probe_log)

    def _auto_load_model(self):
        yolo_path = ocr_path = None
        for p in DEFAULT_YOLO_MODELS:
            if p.exists():
                yolo_path = str(p)
                break
        for p in DEFAULT_OCR_MODELS:
            if p.exists():
                ocr_path = str(p)
                break
        if not yolo_path:
            self.status_var.set('无可用模型')
            return
        self.status_var.set(f'正在加载模型: {Path(yolo_path).name}...')
        self.root.update()
        def load():
            try:
                config = ModelConfig(yolo_model=Path(yolo_path), ocr_model=Path(ocr_path) if ocr_path else Path(''),
                    classes_file=DEFAULT_CLASSES_FILE, dict_file=DEFAULT_DICT_FILE, skip_ocr=not ocr_path)
                engine = TBJURKNNEngine(config)
                engine.load()
                self.engine = engine
                self.status_var.set('模型已加载，就绪')
                self._log('模型加载成功')
            except Exception as e:
                self.status_var.set(f'模型加载失败: {e}')
                self._log(f'模型加载失败: {e}')
        threading.Thread(target=load, daemon=True).start()

    def _open_image(self):
        path = filedialog.askopenfilename(filetypes=[('图片', '*.jpg *.jpeg *.png *.bmp')])
        if path and self.engine:
            self._start_inference('image', path)
        elif not self.engine:
            messagebox.showwarning('提示', '模型未加载')

    def _open_video(self):
        path = filedialog.askopenfilename(filetypes=[
            ('视频文件', '*.mp4 *.avi *.mkv *.mov *.MP4 *.AVI *.MKV *.MOV'),
            ('所有文件', '*.*')])
        if path and self.engine:
            self._pending_source = path
            self._pending_mode = 'video'
            self.status_var.set(f'已加载: {Path(path).name}，点"开始"播放')
            self._log(f'视频已加载: {Path(path).name}')
        elif not self.engine:
            messagebox.showwarning('提示', '模型未加载')

    def _open_camera(self):
        if not self.engine:
            messagebox.showwarning('提示', '模型未加载')
            return
        default_device = '/dev/video21' if sys.platform == 'linux' else '0'
        self._start_inference('camera', default_device)

    def _start(self):
        if self.worker and self.worker.is_alive():
            if self.worker.paused:
                self.worker.request_resume()
                self.status_var.set('运行中')
            return
        if hasattr(self, '_pending_source'):
            self._start_inference(self._pending_mode, self._pending_source)

    def _start_inference(self, mode, source):
        self._stop()
        self.engine.config.conf = self.conf_var.get()
        self.engine.config.iou = self.iou_var.get()
        self.engine.config.skip_ocr = self.skip_ocr_var.get()
        self.worker = InferenceWorker(self.engine, self._on_frame, self._log, self._on_error, self._on_finished)
        self.worker.mode = mode
        self.worker.source = source
        self.worker.every_n = self.every_n_var.get()
        self.worker.output_dir = self.output_dir_var.get()
        metrics_dir = Path(self.output_dir_var.get()) / 'logs'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.system_monitor.start(csv_path=metrics_dir / f'metrics_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        self.worker.start()
        self.status_var.set(f'运行中: {mode}')
        self._log(f'开始推理: {mode}')

    def _pause(self):
        if self.worker and self.worker.is_alive():
            if self.worker.paused:
                self.worker.request_resume()
                self.status_var.set('运行中')
            else:
                self.worker.request_pause()
                self.status_var.set('已暂停')

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.request_stop()
            self.worker.join(timeout=3)
        self._recording = False
        self.record_label.config(text='⚫ 未录制')
        self.system_monitor.stop()
        self.status_var.set('就绪')

    def _toggle_recording(self):
        if not self._recording:
            self._recording = True
            self.btn_record.config(text='停止录制')
            self.record_label.config(text='🔴 录制中')
            if self.worker and self.worker.is_alive():
                self.worker.request_start_recording()
            self._log('开始录制')
        else:
            self._recording = False
            self.btn_record.config(text='开始录制')
            self.record_label.config(text='⚫ 未录制')
            if self.worker and self.worker.is_alive():
                self.worker.request_stop_recording()
            self._log('停止录制')

    def _export_results(self):
        if not self._last_session_dir:
            messagebox.showinfo('提示', '暂无结果可导出')
            return
        session_path = Path(self._last_session_dir)
        if not session_path.exists():
            messagebox.showwarning('提示', f'结果目录不存在:\n{session_path}')
            return
        files = []
        for f in session_path.iterdir():
            ext = f.suffix.lower()
            if self.export_csv_var.get() and ext == '.csv':
                files.append(f)
            elif self.export_img_var.get() and ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                files.append(f)
            elif self.export_video_var.get() and ext in ('.mp4', '.avi', '.mkv'):
                files.append(f)
        if not files:
            messagebox.showinfo('提示', '所选格式下没有可导出的文件')
            return
        export_dir = filedialog.askdirectory()
        if not export_dir:
            return
        import shutil
        try:
            target = Path(export_dir) / session_path.name
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(str(f), str(target / f.name))
            self._log(f'已导出 {len(files)} 个文件到: {target}')
            messagebox.showinfo('导出成功', f'已导出 {len(files)} 个文件到:\n{target}')
        except Exception as e:
            self._log(f'导出失败: {e}')

    def _browse_output(self):
        d = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if d:
            self.output_dir_var.set(d)
            self._log(f'输出目录: {d}')

    def _on_frame(self, vis_frame, result):
        if vis_frame is not None:
            self._display_frame(vis_frame)
        if result is not None:
            self._update_stats(result)

    def _on_error(self, msg):
        self._log(f'[ERROR] {msg}')
        self.status_var.set(f'错误: {msg}')

    def _on_finished(self, session_dir):
        self._last_session_dir = session_dir
        self._recording = False
        self.record_label.config(text='⚫ 未录制')
        self.system_monitor.stop()
        self.status_var.set('就绪')

    def _display_frame(self, frame):
        self._last_frame = frame
        # 用 after 避免 tkinter 渲染阻塞
        self.root.after(0, self._do_display, frame)

    def _do_display(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                return
            h, w = rgb.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)
            if nw < 1 or nh < 1:
                return
            resized = cv2.resize(rgb, (nw, nh))
            from PIL import Image, ImageTk
            img = Image.fromarray(resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.delete('all')
            self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=imgtk)
        except Exception:
            pass

    def _update_stats(self, result):
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
        lines.append(f'\nYOLO: {result.yolo_ms:.1f}ms  OCR: {result.ocr_ms:.1f}ms')
        if result.fps > 0:
            lines.append(f'FPS: {result.fps:.1f}')
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert(tk.END, '\n'.join(lines))
        self.stats_text.config(state=tk.DISABLED)

    def _update_monitor(self):
        self.system_monitor.sample()
        display = self.system_monitor.format_display()
        # 保存滚动位置，更新后恢复，防止自动回弹
        scroll_pos = self.monitor_text.yview()
        self.monitor_text.config(state=tk.NORMAL)
        self.monitor_text.delete('1.0', tk.END)
        self.monitor_text.insert(tk.END, display)
        self.monitor_text.yview_moveto(scroll_pos[0])
        self.monitor_text.config(state=tk.DISABLED)
        self.root.after(1000, self._update_monitor)

    def _log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f'[{ts}] {msg}\n')
        self.log_text.see(tk.END)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
