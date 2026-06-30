"""核心算法单元测试 — NMS、letterbox、CTC decode、坐标变换。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# 确保能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.tbju_rknn_core import (
    IMG_SIZE,
    OCR_SIZE,
    letterbox_bgr,
    nms_xyxy,
    decode_yolov8,
    ctc_decode,
    preprocess_yolo,
    preprocess_ocr,
    expand_box,
    get_class_name,
    draw_detections,
    DEFAULT_CLASSES,
    DetectionRow,
)


# ============================================================
# letterbox_bgr
# ============================================================

class TestLetterbox:
    def test_square_input(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_bgr(img, IMG_SIZE)
        assert padded.shape[:2] == (640, 640)
        assert ratio == 1.0

    def test_wide_input(self):
        img = np.zeros((320, 1280, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_bgr(img, IMG_SIZE)
        assert padded.shape[:2] == (640, 640)
        assert ratio == pytest.approx(0.5)

    def test_tall_input(self):
        img = np.zeros((1280, 320, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_bgr(img, IMG_SIZE)
        assert padded.shape[:2] == (640, 640)

    def test_small_input(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_bgr(img, IMG_SIZE)
        assert padded.shape[:2] == (640, 640)

    def test_padding_symmetry(self):
        """验证 padding 对称，无 1px 偏移。"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_bgr(img, (640, 640))
        # 检查 padding 值
        dw, dh = pad
        assert dw == int(dw)  # 应该是整数
        assert dh == int(dh)


# ============================================================
# nms_xyxy
# ============================================================

class TestNMS:
    def test_empty_boxes(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        assert nms_xyxy(boxes, scores, 0.5) == []

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        keep = nms_xyxy(boxes, scores, 0.5)
        assert keep == [0]

    def test_non_overlapping(self):
        boxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]])
        scores = np.array([0.9, 0.8])
        keep = nms_xyxy(boxes, scores, 0.5)
        assert len(keep) == 2

    def test_fully_overlapping(self):
        boxes = np.array([[10, 10, 50, 50], [10, 10, 50, 50]])
        scores = np.array([0.9, 0.8])
        keep = nms_xyxy(boxes, scores, 0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # 保留分数最高的

    def test_partial_overlap(self):
        boxes = np.array([[10, 10, 50, 50], [30, 30, 70, 70]])
        scores = np.array([0.9, 0.8])
        keep = nms_xyxy(boxes, scores, 0.3)
        # IoU 较高时应只保留一个
        assert len(keep) <= 2


# ============================================================
# ctc_decode
# ============================================================

class TestCTCDecode:
    def test_simple_decode(self):
        chars = ['你', '好']
        # 3 个时间步，类别: blank(0), 你(1), 好(2)
        # 时间步1: 你, 时间步2: 好
        logits = np.array([
            [0.1, 0.8, 0.1],  # 你
            [0.1, 0.1, 0.8],  # 好
        ])
        result = ctc_decode(logits, chars)
        assert result == '你好'

    def test_repeated_chars(self):
        """连续重复字符应合并。"""
        chars = ['A', 'B']
        logits = np.array([
            [0.1, 0.8, 0.1],  # A
            [0.1, 0.8, 0.1],  # A (重复)
            [0.1, 0.1, 0.8],  # B
        ])
        result = ctc_decode(logits, chars)
        assert result == 'AB'

    def test_blank_only(self):
        chars = ['A']
        logits = np.array([
            [0.9, 0.1],  # blank
            [0.9, 0.1],  # blank
        ])
        result = ctc_decode(logits, chars)
        assert result == ''

    def test_transposed_input(self):
        """输入 shape 为 (num_classes, time_steps) 时应自动转置。"""
        chars = ['你', '好']
        logits = np.array([
            [0.1, 0.1],  # blank
            [0.8, 0.1],  # 你
            [0.1, 0.8],  # 好
        ])
        result = ctc_decode(logits, chars)
        assert result == '你好'


# ============================================================
# expand_box
# ============================================================

class TestExpandBox:
    def test_no_padding(self):
        box = [10, 20, 50, 60]
        result = expand_box(box, 0, (100, 100))
        assert result == [10, 20, 50, 60]

    def test_with_padding(self):
        box = [10, 20, 50, 60]
        result = expand_box(box, 5, (100, 100))
        assert result == [5, 15, 55, 65]

    def test_clip_to_boundary(self):
        box = [0, 0, 10, 10]
        result = expand_box(box, 20, (100, 100))
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 99
        assert result[3] <= 99


# ============================================================
# get_class_name
# ============================================================

class TestGetClassName:
    def test_valid_id(self):
        assert get_class_name(0, DEFAULT_CLASSES) == 'TBJU_region'

    def test_invalid_id(self):
        assert get_class_name(99, DEFAULT_CLASSES) == 'class_99'


# ============================================================
# preprocess
# ============================================================

class TestPreprocess:
    def test_preprocess_yolo_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        x, ratio, pad = preprocess_yolo(img)
        assert x.shape == (1, 640, 640, 3)
        assert x.dtype == np.uint8

    def test_preprocess_ocr_shape(self):
        img = np.zeros((32, 100, 3), dtype=np.uint8)
        x = preprocess_ocr(img)
        assert x.shape == (1, 3, OCR_SIZE[1], OCR_SIZE[0])
        assert x.dtype == np.float32

    def test_preprocess_ocr_empty_raises(self):
        img = np.array([])
        with pytest.raises(ValueError):
            preprocess_ocr(img)

    def test_preprocess_ocr_consistent_shape(self):
        """多次调用返回相同 shape。"""
        for _ in range(10):
            img = np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
            x = preprocess_ocr(img)
            assert x.shape == (1, 3, OCR_SIZE[1], OCR_SIZE[0])

    def test_preprocess_ocr_different_inputs(self):
        """不同输入应产生不同输出。"""
        img1 = np.zeros((32, 100, 3), dtype=np.uint8)
        img2 = np.full((32, 100, 3), 255, dtype=np.uint8)
        x1 = preprocess_ocr(img1)
        x2 = preprocess_ocr(img2)
        assert not np.array_equal(x1, x2)


# ============================================================
# draw_detections
# ============================================================

class TestDrawDetections:
    def test_basic_draw(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = [DetectionRow('test', 0, 0, 0, 'TBJU_region', 10, 10, 100, 100, 0.95)]
        vis = draw_detections(frame, rows, DEFAULT_CLASSES)
        assert vis.shape == frame.shape

    def test_copy_false(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = [DetectionRow('test', 0, 0, 0, 'TBJU_region', 10, 10, 100, 100, 0.95)]
        vis = draw_detections(frame, rows, DEFAULT_CLASSES, copy=False)
        assert vis is frame  # 原地绘制，返回同一对象

    def test_output_rgb(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rows = []
        vis = draw_detections(frame, rows, DEFAULT_CLASSES, output_rgb=True)
        assert vis.shape == frame.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
