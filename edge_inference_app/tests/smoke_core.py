#!/usr/bin/env python3
"""smoke_core.py — 推理核心冒烟测试（PC 端，PyTorch 后端）"""

import sys
from pathlib import Path

# 添加项目根目录到 path
rknn_deploy = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(rknn_deploy))

from src.core.tbju_rknn_core import (
    ModelConfig, TBJURKNNEngine, ResultWriter,
    load_chars, load_classes, preprocess_yolo, preprocess_ocr,
)


def test_load_functions():
    print("=== 测试加载函数 ===")
    classes = load_classes(rknn_deploy / 'config' / 'merged_classes.txt')
    print(f"classes: {classes}")
    assert len(classes) == 6, f"期望 6 类，实际 {len(classes)}"

    chars = load_chars(rknn_deploy / 'config' / 'ppocr_keys_v1.txt')
    print(f"chars: {''.join(chars)}")
    assert len(chars) == 15, f"期望 15 字符，实际 {len(chars)}"
    print("OK\n")


def test_preprocess():
    import cv2
    import numpy as np
    print("=== 测试预处理 ===")
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    yolo_input, ratio, pad = preprocess_yolo(dummy_img)
    print(f"YOLO input shape: {yolo_input.shape}, dtype: {yolo_input.dtype}")
    assert yolo_input.shape == (1, 640, 640, 3), f"YOLO shape 错误: {yolo_input.shape}"
    assert yolo_input.dtype == np.uint8

    dummy_crop = np.random.randint(0, 255, (32, 120, 3), dtype=np.uint8)
    ocr_input = preprocess_ocr(dummy_crop)
    print(f"OCR input shape: {ocr_input.shape}, dtype: {ocr_input.dtype}")
    assert ocr_input.shape == (1, 3, 32, 384), f"OCR shape 错误: {ocr_input.shape}"
    print("OK\n")


def test_pytorch_engine():
    print("=== 测试 PyTorch 引擎 ===")
    # 尝试多个可能的 Carriage 路径
    candidates = [
        rknn_deploy.parent / 'Carriage',
        Path(r'C:\Users\LENOVO\Desktop\Carriage'),
    ]
    carriage_root = None
    for c in candidates:
        if (c / 'YOLO_train' / 'run_merged' / 'train' / 'weights' / 'best.pt').exists():
            carriage_root = c
            break
    if carriage_root is None:
        print("SKIP: 找不到 Carriage 目录")
        return
    yolo_path = carriage_root / 'YOLO_train' / 'run_merged' / 'train' / 'weights' / 'best.pt'
    ocr_path = carriage_root / 'OCR_train' / 'output' / 'ppocr_rec_carriage_number' / 'best_model.pth'
    dict_path = carriage_root / 'OCR_train' / 'ppocr_keys_v1.txt'
    classes_path = rknn_deploy / 'config' / 'merged_classes.txt'

    if not yolo_path.exists():
        print(f"SKIP: YOLO 模型不存在: {yolo_path}")
        return
    if not ocr_path.exists():
        print(f"SKIP: OCR 模型不存在: {ocr_path}")
        return

    config = ModelConfig(
        yolo_model=yolo_path, ocr_model=ocr_path,
        classes_file=classes_path, dict_file=dict_path,
    )
    engine = TBJURKNNEngine(config)
    engine.load()
    print("模型加载成功")

    import numpy as np
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.infer_frame(dummy_img, 'test.jpg', 0)
    print(f"推理完成: {len(result.rows)} 个检测, YOLO {result.yolo_ms:.1f}ms, OCR {result.ocr_ms:.1f}ms")

    vis = engine.draw(dummy_img, result)
    print(f"绘制完成: {vis.shape}")

    engine.close()
    print("OK\n")


def test_result_writer():
    print("=== 测试 ResultWriter ===")
    from src.core.tbju_rknn_core import DetectionRow
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ResultWriter(Path(tmpdir), 'test.csv')
        writer.open_csv()
        writer.write_rows([
            DetectionRow(source='test.jpg', frame=0, index=0,
                         class_id=0, class_name='TBJU_region',
                         x1=100, y1=200, x2=300, y2=250,
                         det_conf=0.95, ocr_text='TBJU6970527'),
        ], yolo_ms=50.0, ocr_ms=10.0, total_ms=60.0)
        writer.close()

        csv_path = Path(tmpdir) / 'test.csv'
        assert csv_path.exists(), "CSV 文件未创建"
        content = csv_path.read_text()
        assert 'TBJU6970527' in content, "CSV 内容缺失"
        print(f"CSV 内容:\n{content}")
    print("OK\n")


if __name__ == '__main__':
    test_load_functions()
    test_preprocess()
    test_result_writer()
    test_pytorch_engine()
    print("=== 所有测试通过 ===")
