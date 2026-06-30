#!/usr/bin/env python3
"""
tbju_capacity_test.py — 扩展能力压测模式
模拟 1/2/4 路检测负载，生成 capacity_report。
支持停止、进度回调、metrics CSV、markdown 摘要。
"""

import csv
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.core.tbju_rknn_core import ModelConfig, TBJURKNNEngine, DetectionRow, FrameResult


@dataclass
class CapacityProfile:
    name: str
    roi_count: int
    width: int = 640
    height: int = 480
    every_n: int = 2
    duration_s: int = 30


@dataclass
class CapacityResult:
    profile_name: str
    duration_s: float
    input_resolution: str
    roi_count: int
    every_n: int
    model_name: str
    avg_display_fps: float
    avg_infer_fps: float
    min_display_fps: float
    p50_total_ms: float
    p95_total_ms: float
    avg_yolo_ms: float
    avg_ocr_ms: float
    avg_cpu_percent: float
    peak_memory_mb: float
    max_temp_c: float
    avg_npu_load_percent: float
    avg_gpu_load_percent: float
    estimated_tops: float
    recommendation: str


DEFAULT_PROFILES = [
    CapacityProfile('profile_1way_640_every2', roi_count=1, every_n=2, duration_s=30),
    CapacityProfile('profile_2roi_640_every2', roi_count=2, every_n=2, duration_s=30),
    CapacityProfile('profile_4roi_640_every2', roi_count=4, every_n=2, duration_s=30),
    CapacityProfile('profile_4roi_640_every3', roi_count=4, every_n=3, duration_s=30),
]

RK3588_NPU_TOPS = 6.0


class CapacityTestRunner:
    """可停止的容量压测运行器。"""

    def __init__(self, engine: TBJURKNNEngine, video_source,
                 profiles=None, output_dir=None, monitor=None,
                 progress_callback=None, metrics_callback=None):
        self.engine = engine
        self.video_source = video_source
        self.profiles = profiles or DEFAULT_PROFILES
        self.output_dir = Path(output_dir) if output_dir else Path('output/capacity')
        self.monitor = monitor
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        self._stop_event = threading.Event()
        self._results: List[CapacityResult] = []
        self._session_dir = None
        self._metrics_rows = []
        self._metrics_csv_file = None
        self._metrics_csv_writer = None

    def request_stop(self):
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def run(self) -> List[CapacityResult]:
        self._session_dir = self.output_dir / time.strftime('%Y%m%d_%H%M%S')
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._results = []
        self._metrics_rows = []

        # 初始化 metrics 流式写入
        metrics_path = self._session_dir / 'metrics_during_capacity.csv'
        self._metrics_csv_file = open(metrics_path, 'w', newline='', encoding='utf-8')

        try:
            for i, profile in enumerate(self.profiles):
                if self._stop_event.is_set():
                    if self.progress_callback:
                        self.progress_callback('压测已停止')
                    break
                if self.progress_callback:
                    self.progress_callback(f'运行 profile {i+1}/{len(self.profiles)}: {profile.name}')
                result = self._run_single_profile(profile)
                self._results.append(result)
        finally:
            # 确保 CSV 文件句柄一定被关闭
            if self._metrics_csv_file and not self._metrics_csv_file.closed:
                self._metrics_csv_file.close()
            self._metrics_csv_file = None
            self._metrics_csv_writer = None

        if self._results:
            _save_report(self._results, self._session_dir, self._metrics_rows)
        return self._results

    def _run_single_profile(self, profile: CapacityProfile) -> CapacityResult:
        cap = cv2.VideoCapture(str(self.video_source))
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频源: {self.video_source}')

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError('无法读取首帧')

        h, w = frame.shape[:2]
        roi_regions = _generate_roi_regions(w, h, profile.roi_count)

        total_ms_list = []
        yolo_ms_list = []
        ocr_ms_list = []
        cpu_list = []
        mem_list = []
        temp_list = []
        npu_list = []
        gpu_list = []
        display_fps_list = []

        start_time = time.time()
        frame_id = 0
        infer_count = 0
        last_infer_time = start_time

        while time.time() - start_time < profile.duration_s:
            if self._stop_event.is_set():
                break

            if frame_id % profile.every_n == 0:
                frame_start = time.time()

                if profile.roi_count <= 1:
                    result = self.engine.infer_frame(frame, 'capacity_test', frame_id)
                else:
                    result = _infer_multi_roi(self.engine, frame, roi_regions, 'capacity_test', frame_id)

                frame_end = time.time()
                total_ms = (frame_end - frame_start) * 1000
                total_ms_list.append(total_ms)
                yolo_ms_list.append(result.yolo_ms)
                ocr_ms_list.append(result.ocr_ms)
                infer_count += 1

                # 计算瞬时 FPS
                dt = frame_end - last_infer_time
                if dt > 0:
                    display_fps_list.append(1.0 / dt)
                last_infer_time = frame_end

                if self.monitor:
                    stats = self.monitor.latest()
                    cpu_list.append(stats.cpu_percent)
                    mem_list.append(stats.memory_used_mb)
                    max_temp = max(stats.temp_zones.values()) if stats.temp_zones else 0
                    temp_list.append(max_temp)
                    if stats.npu_load_percent >= 0:
                        npu_list.append(stats.npu_load_percent)
                    if stats.gpu_load_percent >= 0:
                        gpu_list.append(stats.gpu_load_percent)

                    # 记录 metrics 行（内存 + 流式写入磁盘）
                    row = {
                        'timestamp': stats.timestamp,
                        'profile': profile.name,
                        'frame_id': frame_id,
                        'total_ms': round(total_ms, 1),
                        'yolo_ms': round(result.yolo_ms, 1),
                        'ocr_ms': round(result.ocr_ms, 1),
                        'cpu_percent': round(stats.cpu_percent, 1),
                        'memory_mb': round(stats.memory_used_mb, 0),
                        'temp_c': round(max_temp, 1),
                        'npu_load': round(stats.npu_load_percent, 1) if stats.npu_load_percent >= 0 else '',
                        'gpu_load': round(stats.gpu_load_percent, 1) if stats.gpu_load_percent >= 0 else '',
                    }
                    self._metrics_rows.append(row)
                    # 流式写入磁盘
                    if self._metrics_csv_file and not self._metrics_csv_file.closed:
                        if self._metrics_csv_writer is None:
                            self._metrics_csv_writer = csv.DictWriter(
                                self._metrics_csv_file, fieldnames=row.keys())
                            self._metrics_csv_writer.writeheader()
                        self._metrics_csv_writer.writerow(row)
                        if len(self._metrics_rows) % 50 == 0:
                            self._metrics_csv_file.flush()

                if self.metrics_callback:
                    self.metrics_callback({
                        'profile': profile.name,
                        'frame_id': frame_id,
                        'total_ms': total_ms,
                        'yolo_ms': result.yolo_ms,
                        'ocr_ms': result.ocr_ms,
                    })

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            frame_id += 1

        cap.release()

        elapsed = time.time() - start_time
        avg_infer_fps = infer_count / max(elapsed, 1e-6)
        avg_display_fps = frame_id / max(elapsed, 1e-6)

        if total_ms_list:
            sorted_ms = sorted(total_ms_list)
            p50_idx = int(len(sorted_ms) * 0.5)
            p95_idx = int(len(sorted_ms) * 0.95)
            p50 = sorted_ms[p50_idx]
            p95 = sorted_ms[min(p95_idx, len(sorted_ms) - 1)]
        else:
            p50 = p95 = 0

        min_fps = round(min(display_fps_list), 1) if display_fps_list else 0
        avg_npu = round(np.mean(npu_list), 1) if npu_list else 0
        avg_gpu = round(np.mean(gpu_list), 1) if gpu_list else 0
        estimated_tops = round(RK3588_NPU_TOPS * avg_npu / 100, 2) if avg_npu > 0 else 0

        recommendation = _generate_recommendation(
            profile, avg_infer_fps, p50, avg_display_fps,
            max(temp_list) if temp_list else 0
        )

        return CapacityResult(
            profile_name=profile.name,
            duration_s=round(elapsed, 1),
            input_resolution=f'{w}x{h}',
            roi_count=profile.roi_count,
            every_n=profile.every_n,
            model_name=str(self.engine.config.yolo_model.name),
            avg_display_fps=round(avg_display_fps, 1),
            avg_infer_fps=round(avg_infer_fps, 1),
            min_display_fps=min_fps,
            p50_total_ms=round(p50, 1),
            p95_total_ms=round(p95, 1),
            avg_yolo_ms=round(np.mean(yolo_ms_list), 1) if yolo_ms_list else 0,
            avg_ocr_ms=round(np.mean(ocr_ms_list), 1) if ocr_ms_list else 0,
            avg_cpu_percent=round(np.mean(cpu_list), 1) if cpu_list else 0,
            peak_memory_mb=round(max(mem_list), 0) if mem_list else 0,
            max_temp_c=round(max(temp_list), 1) if temp_list else 0,
            avg_npu_load_percent=avg_npu,
            avg_gpu_load_percent=avg_gpu,
            estimated_tops=estimated_tops,
            recommendation=recommendation,
        )


def _generate_roi_regions(frame_w, frame_h, roi_count):
    if roi_count <= 1:
        return []
    regions = []
    roi_w = frame_w // roi_count
    for i in range(roi_count):
        x1 = i * roi_w
        x2 = (i + 1) * roi_w
        regions.append((x1, 0, x2, frame_h))
    return regions


def _infer_multi_roi(engine, frame, roi_regions, source, frame_id):
    all_rows = []
    total_yolo_ms = 0
    total_ocr_ms = 0

    for idx, (x1, y1, x2, y2) in enumerate(roi_regions):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = engine.infer_frame(crop, f'{source}_roi{idx}', frame_id)
        for row in result.rows:
            row.x1 += x1
            row.y1 += y1
            row.x2 += x1
            row.y2 += y1
            all_rows.append(row)
        total_yolo_ms += result.yolo_ms
        total_ocr_ms += result.ocr_ms

    return FrameResult(
        source=source, frame=frame_id, rows=all_rows,
        yolo_ms=total_yolo_ms, ocr_ms=total_ocr_ms,
        total_ms=total_yolo_ms + total_ocr_ms,
    )


def _generate_recommendation(profile, avg_fps, p50_ms, display_fps, max_temp):
    warning_temp = 75
    critical_temp = 85

    if max_temp >= critical_temp:
        return '温度过高，建议降低负载或等待冷却'
    if max_temp >= warning_temp:
        return '温度较高，建议降低 every_n 或分辨率'

    if profile.roi_count == 1:
        if p50_ms < 100:
            return f'单路稳定 (p50={p50_ms:.0f}ms)，可尝试 2 路 ROI'
        elif p50_ms < 200:
            return f'单路可用 (p50={p50_ms:.0f}ms)，2 路需 every_n=3'
        else:
            return f'单路负载较高 (p50={p50_ms:.0f}ms)，建议降低分辨率'
    elif profile.roi_count == 2:
        if avg_fps >= 5:
            return '2 路稳定，可尝试 4 路'
        else:
            return f'2 路 FPS={avg_fps:.1f}，4 路需 every_n=3 或降低分辨率'
    elif profile.roi_count == 4:
        if avg_fps >= 5:
            return '4 路可用，当前参数满足需求'
        elif avg_fps >= 3:
            return f'4 路 FPS={avg_fps:.1f}，建议 every_n=3'
        else:
            return f'4 路 FPS={avg_fps:.1f}，建议降低分辨率或 skip_ocr'

    return '无'


def _save_report(results: List[CapacityResult], session_dir: Path, metrics_rows=None):
    # CSV 报告
    csv_path = session_dir / 'capacity_report.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'profile_name', 'duration_s', 'input_resolution', 'roi_count', 'every_n',
            'model_name', 'avg_display_fps', 'avg_infer_fps', 'min_display_fps',
            'p50_total_ms', 'p95_total_ms', 'avg_yolo_ms', 'avg_ocr_ms',
            'avg_cpu_percent', 'peak_memory_mb', 'max_temp_c',
            'avg_npu_load_percent', 'avg_gpu_load_percent', 'estimated_tops',
            'recommendation',
        ])
        for r in results:
            writer.writerow([
                r.profile_name, f'{r.duration_s:.1f}', r.input_resolution,
                r.roi_count, r.every_n, r.model_name,
                r.avg_display_fps, r.avg_infer_fps, r.min_display_fps,
                r.p50_total_ms, r.p95_total_ms, r.avg_yolo_ms, r.avg_ocr_ms,
                r.avg_cpu_percent, r.peak_memory_mb, r.max_temp_c,
                r.avg_npu_load_percent, r.avg_gpu_load_percent, r.estimated_tops,
                r.recommendation,
            ])

    # JSON 报告
    json_path = session_dir / 'capacity_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([{
            'profile_name': r.profile_name,
            'duration_s': r.duration_s,
            'input_resolution': r.input_resolution,
            'roi_count': r.roi_count,
            'every_n': r.every_n,
            'model_name': r.model_name,
            'avg_display_fps': r.avg_display_fps,
            'avg_infer_fps': r.avg_infer_fps,
            'min_display_fps': r.min_display_fps,
            'p50_total_ms': r.p50_total_ms,
            'p95_total_ms': r.p95_total_ms,
            'avg_yolo_ms': r.avg_yolo_ms,
            'avg_ocr_ms': r.avg_ocr_ms,
            'avg_cpu_percent': r.avg_cpu_percent,
            'peak_memory_mb': r.peak_memory_mb,
            'max_temp_c': r.max_temp_c,
            'avg_npu_load_percent': r.avg_npu_load_percent,
            'avg_gpu_load_percent': r.avg_gpu_load_percent,
            'estimated_tops': r.estimated_tops,
            'recommendation': r.recommendation,
        } for r in results], f, indent=2, ensure_ascii=False)

    # Markdown 摘要
    md_path = session_dir / 'capacity_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 扩展能力评估报告\n\n')
        f.write(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'RK3588 NPU 标称算力: {RK3588_NPU_TOPS} TOPS (INT8)\n\n')
        f.write('## 评估结果\n\n')
        f.write('| Profile | 路数 | every_n | 平均FPS | P50延迟 | P95延迟 | CPU | 内存 | 温度 | NPU | 估算TOPS | 建议 |\n')
        f.write('|---------|------|---------|---------|---------|---------|-----|------|------|-----|----------|------|\n')
        for r in results:
            f.write(f'| {r.profile_name} | {r.roi_count} | {r.every_n} '
                    f'| {r.avg_infer_fps:.1f} | {r.p50_total_ms:.0f}ms | {r.p95_total_ms:.0f}ms '
                    f'| {r.avg_cpu_percent:.0f}% | {r.peak_memory_mb:.0f}MB | {r.max_temp_c:.1f}C '
                    f'| {r.avg_npu_load_percent:.0f}% | {r.estimated_tops:.2f} | {r.recommendation} |\n')
        f.write('\n## 说明\n\n')
        f.write('- 估算 TOPS = 6.0 * NPU_load_percent / 100，为粗略估算，非严格 profiler 测试\n')
        f.write('- min_display_fps 为推理帧间瞬时 FPS 的最小值\n')
        f.write('- 温度过高 (>85C) 时建议降低负载或等待冷却\n')

    # metrics CSV
    if metrics_rows:
        metrics_path = session_dir / 'metrics_during_capacity.csv'
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            if metrics_rows:
                writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
                writer.writeheader()
                writer.writerows(metrics_rows)


# 保留旧接口兼容
def run_capacity_test(engine, video_source, profiles=None,
                      output_dir=None, monitor=None, progress_callback=None):
    runner = CapacityTestRunner(engine, video_source, profiles, output_dir, monitor, progress_callback)
    return runner.run()
