"""
路径配置文件 - 所有路径自动从项目根目录推导，无需手动修改。
"""

import os
from pathlib import Path

# ============================================================
# 项目根目录（自动推导）
# ============================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================
# 数据目录
# ============================================================

# 原始数据（图片 + Label Studio JSON）
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"

# 转换后的数据集（YOLO 训练用）
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = DATASETS_DIR / "output"
TEST_OUTPUT_DIR = DATASETS_DIR / "test_output"

# ============================================================
# 训练输出目录
# ============================================================

YOLO_OUTPUT_DIR = PROJECT_ROOT / "YOLO_train" / "runs"
OCR_OUTPUT_DIR = PROJECT_ROOT / "OCR_train" / "output"
OCR_MODEL_DIR = OCR_OUTPUT_DIR / "ppocr_rec_carriage_number"

# ============================================================
# 脚本目录
# ============================================================

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TEST_DIR = PROJECT_ROOT / "test_model"

# ============================================================
# 数据集路径
# ============================================================

# 车号检测数据集
YOLO_DATASET_EYE = OUTPUT_DIR / "wagon_number_detection_平视"
YOLO_DATASET_SIDE = OUTPUT_DIR / "wagon_number_detection_侧视"
YOLO_DATASET_MERGED = OUTPUT_DIR / "wagon_number_detection_merged"

# 车号 OCR 数据集
OCR_DATASET_EYE = OUTPUT_DIR / "wagon_number_ocr_平视"
OCR_DATASET_SIDE = OUTPUT_DIR / "wagon_number_ocr_侧视"
OCR_DATASET_MERGED = OUTPUT_DIR / "wagon_number_ocr_merged"

# 异物/车门数据集
DEBRIS_CARRIAGE_RIM = OUTPUT_DIR / "carriage_rim_debris_detection"
DEBRIS_TRACK_INTRUSION = OUTPUT_DIR / "track_intrusion_detection"
DOOR_STATE = OUTPUT_DIR / "door_state_detection"

# 测试数据集
TEST_YOLO_DIR = TEST_OUTPUT_DIR / "wagon_number_detection_test"
TEST_OCR_DIR = TEST_OUTPUT_DIR / "wagon_number_ocr_test"

# 测试结果输出目录
TEST_RESULTS_DIR = TEST_DIR / "results"

# 原始测试数据源路径（图片）
TEST_EYE_LEVEL_DIR = RAW_DATA_DIR / "eye_level" / "test"
TEST_SIDE_LEVEL_DIR = RAW_DATA_DIR / "side_view" / "test"

# ============================================================
# 异物素材路径
# ============================================================

DEBRIS_MATERIALS_DIR = RAW_DATA_DIR / "debris_materials"


# ============================================================
# 验证与打印
# ============================================================

def validate_paths():
    """验证关键路径是否存在"""
    errors = []
    checks = [
        (RAW_DATA_DIR, "原始数据目录"),
        (OUTPUT_DIR, "输出目录"),
        (PROJECT_ROOT / "YOLO_train", "YOLO_train 目录"),
        (PROJECT_ROOT / "OCR_train", "OCR_train 目录"),
    ]
    for path, desc in checks:
        if not path.exists():
            errors.append(f"{desc}不存在: {path}")
    return errors


def print_config():
    """打印当前配置"""
    print("=" * 60)
    print("当前路径配置")
    print("=" * 60)
    print(f"项目根目录:   {PROJECT_ROOT}")
    print(f"原始数据目录: {RAW_DATA_DIR}")
    print(f"数据集目录:   {DATASETS_DIR}")
    print(f"输出目录:     {OUTPUT_DIR}")
    print(f"脚本目录:     {SCRIPTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()

    errors = validate_paths()
    if errors:
        print("\n路径验证失败:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n路径验证通过!")
