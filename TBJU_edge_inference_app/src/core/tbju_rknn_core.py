#!/usr/bin/env python3
"""
tbju_rknn_core.py — 推理核心模块
从 inference_tbju.py / inference_tbju_stream.py 抽取，GUI 和 CLI 共用。
不依赖 GUI 框架。
"""

import csv
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# 常量
# ============================================================

IMG_SIZE = (640, 640)  # width, height
OCR_SIZE = (384, 32)   # width, height

# 语义类别名称（按名称解析，不依赖固定 ID）
TBJU_CLASS_NAME = 'TBJU_region'

DEFAULT_CLASSES = [
    'TBJU_region',
    'carriage_rim_region',
    'carriage_rim_debris',
    'track_region',
    'track_intrusion_debris',
    'door_region',
]

CLASS_COLORS = {
    0: (0, 220, 0),
    1: (255, 180, 0),
    2: (0, 128, 255),
    3: (255, 0, 160),
    4: (0, 0, 255),
    5: (180, 80, 255),
}

# 异物→区域 语义映射（按类名，不依赖固定 ID）
# 启动时由 resolve_class_ids() 解析为 {debris_id: region_id}
DEBRIS_REGION_NAMES = {
    'carriage_rim_debris': 'carriage_rim_region',
    'track_intrusion_debris': 'track_region',
}

# 时序一致性参数
TEMPORAL_WINDOW_SIZE = 5   # 滑动窗口帧数
TEMPORAL_MIN_HITS = 3      # 窗口内最少出现次数才确认


def resolve_debris_ids(classes: List[str]) -> set:
    """
    从类名列表中提取异物类 ID 集合。不校验区域类别。
    用于时序过滤（只需知道哪些是异物类）。
    """
    name_to_id = {name: i for i, name in enumerate(classes)}
    debris_ids = set()
    for debris_name in DEBRIS_REGION_NAMES:
        if debris_name in name_to_id:
            debris_ids.add(name_to_id[debris_name])
    return debris_ids


def resolve_class_ids(classes: List[str]) -> Dict[int, int]:
    """
    由类名列表构建异物→区域 ID 映射，并校验所有必需类别存在。
    仅在启用区域过滤时调用。
    Args:
        classes: 按 ID 索引排列的类名列表
    Returns:
        {debris_class_id: region_class_id}
    Raises:
        ValueError: 缺少必需类别
    """
    name_to_id = {name: i for i, name in enumerate(classes)}
    missing = []
    for debris_name, region_name in DEBRIS_REGION_NAMES.items():
        if debris_name not in name_to_id:
            missing.append(debris_name)
        if region_name not in name_to_id:
            missing.append(region_name)
    if missing:
        raise ValueError(
            f'模型类别文件缺少必需类别: {missing}，'
            f'实际类别: {classes}'
        )
    return {
        name_to_id[dname]: name_to_id[rname]
        for dname, rname in DEBRIS_REGION_NAMES.items()
    }


# ============================================================
# 区域先验 + 时序一致性 — 后处理
# ============================================================

def _bbox_area(box):
    """计算 bbox 面积 [x1,y1,x2,y2]"""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(a, b):
    """两个 bbox 的交集面积"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def validate_debris_region(dets, debris_region_map: Dict[int, int],
                           containment_thresh=0.3):
    """
    区域验证：检查异物是否落在其对应区域内，不修改模型预测的类别。

    逻辑：
    1. 异物落在其对应区域内 → 通过（保留原始类别）
    2. 对应区域未检测到 → 通过（区域缺失时不丢弃异物，避免漏报）
    3. 对应区域检测到但异物不在其中 → 过滤（背景误报）
    4. 区域类、车号、车门 → 直接通过

    Args:
        dets: decode_yolov8 输出的 dict list, 每个含 bbox/score/class_id
        debris_region_map: {debris_class_id: region_class_id} 动态映射
        containment_thresh: 交集面积占异物面积的比例阈值
    Returns:
        验证后的 dets list
    """
    region_ids = set(debris_region_map.values())

    # 收集所有区域 bbox
    region_boxes = {}  # region_class_id -> list of bboxes
    for d in dets:
        if d['class_id'] in region_ids:
            region_boxes.setdefault(d['class_id'], []).append(d['bbox'])

    filtered = []
    for d in dets:
        cid = d['class_id']
        # 非异物类直接通过
        if cid not in debris_region_map:
            filtered.append(d)
            continue

        # 异物类：验证是否在对应区域内
        expected_region = debris_region_map[cid]
        regions = region_boxes.get(expected_region, [])

        # 对应区域未检测到 → 保留异物（区域缺失时不丢弃）
        if not regions:
            filtered.append(d)
            continue

        # 对应区域已检测到 → 检查重叠
        debris_area = _bbox_area(d['bbox'])
        if debris_area <= 0:
            continue

        valid = False
        for rbox in regions:
            inter = _intersection_area(d['bbox'], rbox)
            if inter / debris_area >= containment_thresh:
                valid = True
                break

        if valid:
            filtered.append(d)
        # else: 对应区域内无足够重叠 → 过滤（背景误报）

    return filtered


class TemporalConsistencyFilter:
    """
    时序一致性：异物在滑动窗口内至少出现 M 次才确认告警。
    区域类和车号类不做过滤，直接通过。
    帧间匹配基于 bbox 中心距离（同一异物在相邻帧中位置应接近）。
    按 source 隔离历史，避免不同视频源互相续帧。
    """

    def __init__(self, window_size=TEMPORAL_WINDOW_SIZE,
                 min_hits=TEMPORAL_MIN_HITS, match_ratio=0.5,
                 min_match_pixels=10.0):
        if window_size < 1:
            raise ValueError(f'window_size must be >= 1, got {window_size}')
        if not (1 <= min_hits <= window_size):
            raise ValueError(f'min_hits must be in [1, window_size], got {min_hits}')
        if match_ratio <= 0:
            raise ValueError(f'match_ratio must be > 0, got {match_ratio}')
        self.window_size = window_size
        self.min_hits = min_hits
        self.match_ratio = match_ratio  # 匹配距离 = 框对角线 × ratio
        self.min_match_pixels = min_match_pixels  # 最小匹配像素下限
        # 按 source 隔离：source -> {class_id: deque of [center_list_per_frame]}
        self._source_histories: Dict[str, Dict[int, deque]] = {}

    def reset(self, source: str = None):
        """重置指定 source 或全部的历史。"""
        if source is not None:
            self._source_histories.pop(source, None)
        else:
            self._source_histories.clear()

    @staticmethod
    def _bbox_center_and_diag(box):
        """返回 (中心点, 对角线长度)。"""
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        diag = (w * w + h * h) ** 0.5
        return (cx, cy), diag

    def update_and_filter(self, dets, source: str = 'default',
                          debris_ids: set = None):
        """
        输入当前帧的 dets（已经过区域验证），输出确认的 dets。
        区域类/车号类直接通过，异物类需要时序确认。
        每帧都会更新所有已知异物类的历史（无检测时记录空列表）。
        帧间匹配基于 bbox 中心距离，阈值 = 框对角线 × match_ratio（自适应分辨率）。

        Args:
            dets: 当前帧检测结果
            source: 视频源标识，用于隔离历史
            debris_ids: 异物类 ID 集合，用于区分异物和非异物
        """
        if debris_ids is None:
            raise ValueError('debris_ids must be provided as a set of integer class IDs')

        # 获取该 source 的历史
        if source not in self._source_histories:
            self._source_histories[source] = {}
        history_map = self._source_histories[source]

        # 分离异物和非异物
        non_debris = []
        debris_by_class = {}  # class_id -> list of (idx, det)
        for i, d in enumerate(dets):
            cid = d['class_id']
            if cid in debris_ids:
                debris_by_class.setdefault(cid, []).append((i, d))
            else:
                non_debris.append(d)

        confirmed = list(non_debris)

        # 遍历所有已知异物类（包括当前帧无检测但历史中有的类）
        all_cids = set(history_map.keys()) | set(debris_by_class.keys())
        for cid in all_cids:
            items = debris_by_class.get(cid, [])
            # 当前帧数据：(center, diag) 元组
            curr_data = [
                self._bbox_center_and_diag(d['bbox']) for _, d in items
            ]

            # 获取或创建历史
            if cid not in history_map:
                history_map[cid] = deque(maxlen=self.window_size)
            history = history_map[cid]

            # 记录当前帧（无检测时记录空列表，确保旧帧被滑出）
            history.append(curr_data)

            # 只有当前帧有检测时才做确认判断
            if not curr_data:
                continue

            # 统计每个当前检测在窗口内的命中次数
            hit_counts = [1] * len(curr_data)  # 当前帧算一次

            for past_idx in range(len(history) - 1):
                past_data = history[past_idx]
                if not past_data:
                    continue

                # 贪心一对一匹配：先算距离矩阵，再贪心分配
                # 距离矩阵：curr[i] → past[j]
                n_curr = len(curr_data)
                n_past = len(past_data)
                matches = []  # (dist, ci, pi)
                for ci, (cc, c_diag) in enumerate(curr_data):
                    for pi, (pc, p_diag) in enumerate(past_data):
                        dist = ((cc[0] - pc[0]) ** 2 + (cc[1] - pc[1]) ** 2) ** 0.5
                        thresh = max(
                            (c_diag + p_diag) / 2.0 * self.match_ratio,
                            self.min_match_pixels,
                        )
                        if dist <= thresh:
                            matches.append((dist, ci, pi))

                # 贪心分配：按距离排序，每次分配一对
                matches.sort()
                assigned_curr = set()
                assigned_past = set()
                for dist, ci, pi in matches:
                    if ci not in assigned_curr and pi not in assigned_past:
                        hit_counts[ci] += 1
                        assigned_curr.add(ci)
                        assigned_past.add(pi)

            for ci in range(len(curr_data)):
                if hit_counts[ci] >= self.min_hits:
                    confirmed.append(items[ci][1])

        return confirmed


# ============================================================
# 数据类
# ============================================================

@dataclass
class ModelConfig:
    yolo_model: Path
    ocr_model: Path
    classes_file: Path
    dict_file: Path
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100
    crop_pad: int = 4
    tbju_class_id: int = 0
    skip_ocr: bool = False
    all_cores: bool = True
    enable_region_filter: bool = True       # 区域重分类：根据位置确定异物类别
    enable_temporal_filter: bool = True     # 时序一致性：滑动窗口内命中M次确认
    temporal_window: int = TEMPORAL_WINDOW_SIZE
    temporal_min_hits: int = TEMPORAL_MIN_HITS
    region_containment_thresh: float = 0.3  # 异物与区域重叠比例阈值
    # 由 TBJURKNNEngine.__init__ 自动填充，无需手动设置
    debris_region_map: Optional[Dict[int, int]] = field(default=None, repr=False)


@dataclass
class DetectionRow:
    source: str
    frame: int
    index: int
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    det_conf: float
    ocr_text: str = ""


@dataclass
class FrameResult:
    source: str
    frame: int
    rows: List[DetectionRow]
    yolo_ms: float
    ocr_ms: float
    total_ms: float
    fps: float = 0.0


# ============================================================
# 工具函数
# ============================================================

def load_chars(path) -> List[str]:
    chars = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            s = line.strip()
            if s:
                chars.append(s)
    if not chars:
        raise ValueError(f'empty char dict: {path}')
    return chars


def load_classes(path) -> List[str]:
    """
    加载类别名称列表。支持两种格式：
    - 带显式 ID: "0 TBJU_region\\n1 carriage_rim_region\\n..."
    - 纯名称: "TBJU_region\\ncarriage_rim_region\\n..."
    带 ID 时按 ID 索引填充，校验重复 ID 和缺号。
    """
    if not path or not Path(path).exists():
        return DEFAULT_CLASSES[:]

    entries_with_id = []  # (int_id, name)
    entries_without_id = []  # name
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                entries_with_id.append((int(parts[0]), parts[1]))
            else:
                entries_without_id.append(s)

    # 纯名称格式
    if not entries_with_id:
        if not entries_without_id:
            return DEFAULT_CLASSES[:]
        # 校验重复类名
        seen_names = set()
        for name in entries_without_id:
            if name in seen_names:
                raise ValueError(f'类别文件有重复类名: {name}')
            seen_names.add(name)
        return entries_without_id

    # 带 ID 格式：校验并按索引填充
    if entries_without_id:
        raise ValueError(
            f'类别文件格式混用：部分行有 ID，部分没有: {entries_without_id[:3]}'
        )

    seen_ids = set()
    seen_names = set()
    for cid, name in entries_with_id:
        if cid in seen_ids:
            raise ValueError(f'类别文件有重复 ID: {cid}')
        if name in seen_names:
            raise ValueError(f'类别文件有重复类名: {name}')
        seen_ids.add(cid)
        seen_names.add(name)

    max_id = max(seen_ids)
    result = [None] * (max_id + 1)
    for cid, name in entries_with_id:
        result[cid] = name

    missing = [i for i, v in enumerate(result) if v is None]
    if missing:
        raise ValueError(f'类别文件缺号: {missing}')

    return result


def letterbox_bgr(image, new_shape=IMG_SIZE, color=(0, 0, 0)):
    src_h, src_w = image.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / src_w, new_h / src_h)
    resized_w = int(round(src_w * r))
    resized_h = int(round(src_h * r))
    dw = new_w - resized_w
    dh = new_h - resized_h
    if (src_w, src_h) != (resized_w, resized_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw / 2.0, dh / 2.0)


def nms_xyxy(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    # cv2.dnn.NMSBoxes 要求 [x, y, w, h] 格式
    boxes_xywh = boxes.copy().astype(np.float32)
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # w = x2 - x1
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # h = y2 - y1
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        scores.astype(np.float32).tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_thresh),
    )
    if len(indices) == 0:
        return []
    return indices.flatten().tolist()


def decode_yolov8(outputs, ratio, pad, orig_shape, conf_thresh, iou_thresh, max_det=100):
    pred = np.asarray(outputs[0])
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError(f'unexpected YOLO output shape: {outputs[0].shape}')
    if pred.shape[0] <= 20 and pred.shape[1] > pred.shape[0]:
        pred = pred.T
    if pred.shape[1] < 5:
        raise RuntimeError(f'unexpected YOLO output layout: {pred.shape}')

    boxes_xywh = pred[:, :4].astype(np.float32)
    cls_scores = pred[:, 4:].astype(np.float32)
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    mask = scores >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if len(scores) == 0:
        return []

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    orig_h, orig_w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    dets = []
    for cls_id in sorted(set(class_ids.tolist())):
        idxs = np.where(class_ids == cls_id)[0]
        keep = nms_xyxy(boxes[idxs], scores[idxs], iou_thresh)
        for k in keep:
            i = idxs[k]
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({
                'bbox': [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                'score': float(scores[i]),
                'class_id': int(class_ids[i]),
            })
    dets.sort(key=lambda d: d['score'], reverse=True)
    return dets[:max_det]


def preprocess_yolo(image_bgr):
    """YOLO 预处理 — 返回 NHWC uint8（RKNN 板端格式）"""
    padded, ratio, pad = letterbox_bgr(image_bgr, IMG_SIZE, color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8), ratio, pad


def preprocess_yolo_nchw(image_bgr):
    """YOLO 预处理 — 返回 NCHW float32（PyTorch/ONNX 格式）"""
    padded, ratio, pad = letterbox_bgr(image_bgr, IMG_SIZE, color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0), ratio, pad


def preprocess_ocr(crop_bgr):
    if crop_bgr.size == 0:
        raise ValueError('empty OCR crop')
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, OCR_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32)


def ctc_decode(logits, chars, blank_idx=0):
    arr = np.asarray(logits)
    arr = np.squeeze(arr)
    num_classes = len(chars) + 1
    if arr.ndim != 2:
        raise RuntimeError(f'unexpected OCR output shape: {logits.shape}')
    if arr.shape[-1] != num_classes and arr.shape[0] == num_classes:
        arr = arr.T
    ids = np.argmax(arr, axis=-1).tolist()
    result = []
    prev = None
    for idx in ids:
        if idx != blank_idx and idx != prev and 1 <= idx <= len(chars):
            result.append(chars[idx - 1])
        prev = idx
    return ''.join(result)


def expand_box(box, pad, shape):
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    ]


def get_class_name(class_id, classes):
    if 0 <= class_id < len(classes):
        return classes[class_id]
    return f'class_{class_id}'


def safe_csv_cell(value):
    """防止 CSV 注入：Excel 会把以 = + - @ 开头的单元格当公式执行。"""
    if value is None:
        return ''
    s = str(value)
    if s.startswith(('=', '+', '-', '@')):
        return "'" + s
    return s


def draw_detections(frame, rows, classes, fps_text='', copy=True, output_rgb=False):
    vis = frame.copy() if copy else frame
    img_h, img_w = vis.shape[:2]
    for row in rows:
        cls_id = row.class_id
        name = get_class_name(cls_id, classes)
        x1, y1 = row.x1, row.y1
        x2 = min(row.x2, img_w - 1)
        y2 = min(row.y2, img_h - 1)
        color = CLASS_COLORS.get(cls_id, (0, 220, 0))
        label = f'{name} {row.det_conf:.2f}'
        if row.ocr_text:
            label = f'{row.ocr_text} {row.det_conf:.2f}'
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if fps_text:
        cv2.rectangle(vis, (8, 8), (420, 42), (0, 0, 0), -1)
        cv2.putText(vis, fps_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if output_rgb:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return vis


# ============================================================
# 推理后端抽象
# ============================================================

class BaseInferenceBackend(ABC):
    @abstractmethod
    def load(self, config: 'ModelConfig') -> None: ...

    @abstractmethod
    def infer_frame(self, frame_bgr, source: str, frame_id: int, config: ModelConfig,
                    classes: List[str], chars: List[str]) -> FrameResult: ...

    @abstractmethod
    def close(self) -> None: ...


class RKNNLiteBackend(BaseInferenceBackend):
    """RK3588 端后端 — 使用 rknn-toolkit-lite2 + NPU"""

    def __init__(self):
        self.yolo_runner = None
        self.ocr_runner = None

    def load(self, config: ModelConfig) -> None:
        from rknnlite.api import RKNNLite
        try:
            self.yolo_runner = self._create_runner(str(config.yolo_model), config.all_cores)
            if not config.skip_ocr:
                self.ocr_runner = self._create_runner(str(config.ocr_model), config.all_cores)
        except Exception:
            self.close()
            raise

    def _create_runner(self, model_path, use_all_cores):
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        try:
            ret = rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f'load_rknn failed: {model_path}, ret={ret}')
            if use_all_cores:
                try:
                    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
                except Exception as e:
                    warnings.warn(f'NPU_CORE_0_1_2 unavailable ({e}), falling back to default core')
                    ret = rknn.init_runtime()
            else:
                ret = rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f'init_runtime failed: {model_path}, ret={ret}')
            return rknn
        except Exception:
            try:
                rknn.release()
            except Exception:
                pass
            raise

    def infer_frame(self, frame_bgr, source, frame_id, config, classes, chars):
        t0 = time.time()
        yolo_input, ratio, pad = preprocess_yolo(frame_bgr)
        yolo_outputs = self.yolo_runner.inference(inputs=[yolo_input])
        dets = decode_yolov8(yolo_outputs, ratio, pad, frame_bgr.shape, config.conf, config.iou, config.max_det)
        # 区域先验：过滤不在对应区域内的异物
        if config.enable_region_filter:
            if config.enable_region_filter and config.debris_region_map:
                dets = validate_debris_region(dets, config.debris_region_map, config.region_containment_thresh)
        t1 = time.time()

        rows = []
        for idx, det in enumerate(dets):
            cls_id = det['class_id']
            name = get_class_name(cls_id, classes)
            box = expand_box(det['bbox'], config.crop_pad if cls_id == config.tbju_class_id else 0, frame_bgr.shape)
            x1, y1, x2, y2 = box
            text = ''
            if (not config.skip_ocr) and cls_id == config.tbju_class_id and self.ocr_runner is not None:
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    ocr_input = preprocess_ocr(crop)
                    ocr_outputs = self.ocr_runner.inference(inputs=[ocr_input], data_format='nchw')
                    text = ctc_decode(ocr_outputs[0], chars)
            rows.append(DetectionRow(
                source=source, frame=frame_id, index=idx,
                class_id=cls_id, class_name=name,
                x1=x1, y1=y1, x2=x2, y2=y2,
                det_conf=float(det['score']), ocr_text=text,
            ))
        t2 = time.time()

        return FrameResult(
            source=source, frame=frame_id, rows=rows,
            yolo_ms=(t1 - t0) * 1000.0, ocr_ms=(t2 - t1) * 1000.0,
            total_ms=(t2 - t0) * 1000.0,
        )

    def close(self):
        if self.yolo_runner is not None:
            self.yolo_runner.release()
            self.yolo_runner = None
        if self.ocr_runner is not None:
            self.ocr_runner.release()
            self.ocr_runner = None


class PyTorchBackend(BaseInferenceBackend):
    """PC 端开发后端 — 使用 ultralytics + torch"""

    def __init__(self):
        self.yolo_model = None
        self.ocr_model = None
        self.device = 'cpu'

    def load(self, config: ModelConfig) -> None:
        import torch
        from ultralytics import YOLO
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo_model = YOLO(str(config.yolo_model))
            if not config.skip_ocr:
                import sys
                ocr_model_path = Path(config.ocr_model).resolve()
                candidates = [
                    ocr_model_path.parent.parent.parent,
                    ocr_model_path.parent.parent,
                    ocr_model_path.parent.parent.parent.parent / 'OCR_train',
                    ocr_model_path.parent.parent.parent.parent / 'YOLO_train' / '..' / 'OCR_train',
                    Path(__file__).resolve().parent.parent.parent.parent / 'Carriage' / 'OCR_train',
                ]
                ocr_train_dir = None
                for c in candidates:
                    c = c.resolve()
                    if (c / 'ocr_model.py').exists():
                        ocr_train_dir = c
                        break
                if ocr_train_dir is None:
                    raise FileNotFoundError(f'找不到 OCR_train/ocr_model.py，尝试过: {[str(c) for c in candidates]}')
                if str(ocr_train_dir) not in sys.path:
                    sys.path.insert(0, str(ocr_train_dir))
                from ocr_model import PPoCRRec
                num_classes = len(load_chars(config.dict_file)) + 1
                self.ocr_model = PPoCRRec(num_classes=num_classes)
                self.ocr_model.load_state_dict(
                    torch.load(str(config.ocr_model), map_location=self.device, weights_only=True))
                self.ocr_model.to(self.device)
                self.ocr_model.eval()
        except Exception:
            self.close()
            raise

    def infer_frame(self, frame_bgr, source, frame_id, config, classes, chars):
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        results = self.yolo_model(frame_bgr, conf=config.conf, iou=config.iou,
                                  max_det=config.max_det, imgsz=640, verbose=False)

        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(box.conf[0]),
                'class_id': int(box.cls[0]),
            })
        # 区域先验：过滤不在对应区域内的异物
        if config.enable_region_filter:
            if config.enable_region_filter and config.debris_region_map:
                dets = validate_debris_region(dets, config.debris_region_map, config.region_containment_thresh)
        t1 = time.time()

        rows = []
        for idx, det in enumerate(dets):
            cls_id = det['class_id']
            name = get_class_name(cls_id, classes)
            box = expand_box(det['bbox'], config.crop_pad if cls_id == config.tbju_class_id else 0, frame_bgr.shape)
            x1, y1, x2, y2 = box
            text = ''
            if (not config.skip_ocr) and cls_id == config.tbju_class_id and self.ocr_model is not None:
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    ocr_input = preprocess_ocr(crop)
                    tensor = torch.from_numpy(ocr_input).float().to(self.device)
                    with torch.inference_mode():
                        logits = self.ocr_model(tensor)
                        log_probs = F.log_softmax(logits, dim=-1)
                    text = ctc_decode(log_probs.cpu().numpy()[0], chars)
            rows.append(DetectionRow(
                source=source, frame=frame_id, index=idx,
                class_id=cls_id, class_name=name,
                x1=x1, y1=y1, x2=x2, y2=y2,
                det_conf=float(det['score']), ocr_text=text,
            ))
        t2 = time.time()

        return FrameResult(
            source=source, frame=frame_id, rows=rows,
            yolo_ms=(t1 - t0) * 1000.0, ocr_ms=(t2 - t1) * 1000.0,
            total_ms=(t2 - t0) * 1000.0,
        )

    def close(self):
        self.yolo_model = None
        self.ocr_model = None


class ONNXRuntimeBackend(BaseInferenceBackend):
    """PC 端 ONNX 后端 — 使用 onnxruntime"""

    def __init__(self):
        self.yolo_session = None
        self.ocr_session = None

    def load(self, config: ModelConfig) -> None:
        import onnxruntime as ort
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.yolo_session = ort.InferenceSession(str(config.yolo_model), providers=providers)
            if not config.skip_ocr:
                self.ocr_session = ort.InferenceSession(str(config.ocr_model), providers=providers)
        except Exception:
            self.close()
            raise

    def infer_frame(self, frame_bgr, source, frame_id, config, classes, chars):
        t0 = time.time()
        yolo_input, ratio, pad = preprocess_yolo_nchw(frame_bgr)
        input_name = self.yolo_session.get_inputs()[0].name
        yolo_outputs = self.yolo_session.run(None, {input_name: yolo_input})
        dets = decode_yolov8(yolo_outputs, ratio, pad, frame_bgr.shape, config.conf, config.iou, config.max_det)
        # 区域先验：过滤不在对应区域内的异物
        if config.enable_region_filter:
            if config.enable_region_filter and config.debris_region_map:
                dets = validate_debris_region(dets, config.debris_region_map, config.region_containment_thresh)
        t1 = time.time()

        rows = []
        for idx, det in enumerate(dets):
            cls_id = det['class_id']
            name = get_class_name(cls_id, classes)
            box = expand_box(det['bbox'], config.crop_pad if cls_id == config.tbju_class_id else 0, frame_bgr.shape)
            x1, y1, x2, y2 = box
            text = ''
            if (not config.skip_ocr) and cls_id == config.tbju_class_id and self.ocr_session is not None:
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    ocr_input = preprocess_ocr(crop)
                    ocr_input_name = self.ocr_session.get_inputs()[0].name
                    ocr_outputs = self.ocr_session.run(None, {ocr_input_name: ocr_input})
                    text = ctc_decode(ocr_outputs[0], chars)
            rows.append(DetectionRow(
                source=source, frame=frame_id, index=idx,
                class_id=cls_id, class_name=name,
                x1=x1, y1=y1, x2=x2, y2=y2,
                det_conf=float(det['score']), ocr_text=text,
            ))
        t2 = time.time()

        return FrameResult(
            source=source, frame=frame_id, rows=rows,
            yolo_ms=(t1 - t0) * 1000.0, ocr_ms=(t2 - t1) * 1000.0,
            total_ms=(t2 - t0) * 1000.0,
        )

    def close(self):
        self.yolo_session = None
        self.ocr_session = None


# ============================================================
# 引擎工厂
# ============================================================

def create_backend(config: ModelConfig) -> BaseInferenceBackend:
    """根据模型文件后缀自动选择推理后端。"""
    suffix = config.yolo_model.suffix.lower()

    # .rknn → RKNN 后端
    if suffix == '.rknn':
        try:
            from rknnlite.api import RKNNLite  # noqa: F401
        except ImportError:
            raise RuntimeError('.rknn 模型需要 rknn-toolkit-lite2，请安装后重试')
        try:
            backend = RKNNLiteBackend()
            backend.load(config)
            return backend
        except Exception as e:
            raise RuntimeError(f'RKNN 后端初始化失败: {e}') from e

    # .onnx → ONNX 后端
    if suffix == '.onnx':
        try:
            backend = ONNXRuntimeBackend()
            backend.load(config)
            return backend
        except ImportError:
            raise RuntimeError('ONNX 模型需要 onnxruntime，请执行: pip install onnxruntime')
        except Exception as e:
            raise RuntimeError(f'ONNX 后端初始化失败: {e}') from e

    # .pt / .pth → PyTorch 后端
    try:
        backend = PyTorchBackend()
        backend.load(config)
        return backend
    except ImportError:
        raise RuntimeError(
            '无可用推理后端。请安装以下之一:\n'
            '  - rknn-toolkit-lite2 (RK3588)\n'
            '  - onnxruntime (ONNX 模型)\n'
            '  - ultralytics + torch (PyTorch 模型)'
        )
    except Exception as e:
        raise RuntimeError(f'PyTorch 后端初始化失败: {e}') from e


# ============================================================
# TBJURKNNEngine — 高层封装
# ============================================================

class TBJURKNNEngine:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.classes: List[str] = load_classes(config.classes_file)
        self.chars: List[str] = [] if config.skip_ocr else load_chars(config.dict_file)
        self.backend: Optional[BaseInferenceBackend] = None
        self._infer_lock = threading.Lock()

        # tbju_class_id 按名称解析，不依赖默认值 0
        name_to_id = {name: i for i, name in enumerate(self.classes)}
        if TBJU_CLASS_NAME not in name_to_id:
            raise ValueError(
                f'模型类别缺少车号区域类别 "{TBJU_CLASS_NAME}"，实际类别: {self.classes}'
            )
        self.config.tbju_class_id = name_to_id[TBJU_CLASS_NAME]

        # 异物类 ID（时序过滤需要，不要求区域类存在）
        self._debris_ids: set = resolve_debris_ids(self.classes)

        # 区域映射（仅启用区域过滤时校验区域类）
        if config.enable_region_filter:
            self.config.debris_region_map = resolve_class_ids(self.classes)
        else:
            self.config.debris_region_map = None

        # 时序一致性过滤器
        self._temporal_filter: Optional[TemporalConsistencyFilter] = None
        self._ensure_temporal_filter()

    def _ensure_temporal_filter(self) -> None:
        """按配置创建时序过滤器（如果需要且不存在）。"""
        if self.config.enable_temporal_filter and self._temporal_filter is None:
            self._temporal_filter = TemporalConsistencyFilter(
                window_size=self.config.temporal_window,
                min_hits=self.config.temporal_min_hits,
            )

    def load(self) -> None:
        with self._infer_lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
            self.backend = create_backend(self.config)
            self._ensure_temporal_filter()

    def infer_frame(self, frame_bgr: np.ndarray, source: str, frame_id: int) -> FrameResult:
        with self._infer_lock:
            if self.backend is None:
                raise RuntimeError('engine not loaded')
            result = self.backend.infer_frame(
                frame_bgr, source, frame_id, self.config, self.classes, self.chars)
            # 时序一致性：每帧都调用（即使 rows 为空，确保历史推进）
            if self.config.enable_temporal_filter and self._temporal_filter is not None:
                dets_for_filter = [
                    {'bbox': [r.x1, r.y1, r.x2, r.y2], 'score': r.det_conf, 'class_id': r.class_id}
                    for r in result.rows
                ]
                confirmed_dets = self._temporal_filter.update_and_filter(
                    dets_for_filter, source=source, debris_ids=self._debris_ids,
                )
                confirmed_set = {id(d) for d in confirmed_dets}
                result.rows = [
                    r for r, d in zip(result.rows, dets_for_filter)
                    if id(d) in confirmed_set
                ]
                for i, row in enumerate(result.rows):
                    row.index = i
        return result

    def draw(self, frame_bgr: np.ndarray, result: FrameResult,
             overlay: str = "", output_rgb: bool = False) -> np.ndarray:
        vis = draw_detections(frame_bgr, result.rows, self.classes, overlay, output_rgb=output_rgb)
        return vis

    def close(self) -> None:
        with self._infer_lock:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
            self._temporal_filter = None

    def reset_temporal(self, source: str = None) -> None:
        """重置时序过滤器状态。传 source 只重置该源，不传重置全部。"""
        with self._infer_lock:
            if self._temporal_filter is not None:
                self._temporal_filter.reset(source=source)

    def set_temporal_enabled(self, enabled: bool) -> None:
        """动态启用/禁用时序过滤器。安全创建或释放过滤器实例。"""
        with self._infer_lock:
            self.config.enable_temporal_filter = enabled
            if enabled:
                self._ensure_temporal_filter()
            else:
                self._temporal_filter = None


# ============================================================
# ResultWriter — 结果写入
# ============================================================

class ResultWriter:
    def __init__(self, session_dir: Path, csv_name: str = 'result.csv'):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.session_dir / csv_name
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.video_path = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open_csv(self) -> None:
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'source', 'frame', 'index', 'class_id', 'class_name',
            'x1', 'y1', 'x2', 'y2', 'det_conf', 'ocr_text',
            'yolo_ms', 'ocr_ms', 'total_ms', 'timestamp',
        ])
        self.csv_writer.writeheader()

    def write_rows(self, rows: List[DetectionRow], yolo_ms: float = 0,
                   ocr_ms: float = 0, total_ms: float = 0) -> None:
        if self.csv_writer is None:
            return
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        for row in rows:
            self.csv_writer.writerow({
                'source': row.source, 'frame': row.frame, 'index': row.index,
                'class_id': row.class_id, 'class_name': row.class_name,
                'x1': row.x1, 'y1': row.y1, 'x2': row.x2, 'y2': row.y2,
                'det_conf': row.det_conf, 'ocr_text': safe_csv_cell(row.ocr_text),
                'yolo_ms': f'{yolo_ms:.1f}', 'ocr_ms': f'{ocr_ms:.1f}',
                'total_ms': f'{total_ms:.1f}', 'timestamp': ts,
            })

    def save_image(self, image_bgr: np.ndarray, name: str) -> Path:
        out_path = self.session_dir / name
        cv2.imwrite(str(out_path), image_bgr)
        return out_path

    def open_video(self, output_path: Path, fps: float, width: int, height: int) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mp4_path = output_path.with_suffix('.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
        if self.video_writer.isOpened():
            self.video_path = mp4_path
            return
        avi_path = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(str(avi_path), fourcc, fps, (width, height))
        if self.video_writer.isOpened():
            self.video_path = avi_path
            return
        raise RuntimeError(f'failed to open VideoWriter: {output_path}')

    def write_frame(self, image_bgr: np.ndarray) -> None:
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.write(image_bgr)

    def close(self) -> None:
        if self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
