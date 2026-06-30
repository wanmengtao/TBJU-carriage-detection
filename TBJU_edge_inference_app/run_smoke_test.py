#!/usr/bin/env python3
"""运行冒烟测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.smoke_core import test_load_functions, test_preprocess, test_result_writer, test_pytorch_engine

test_load_functions()
test_preprocess()
test_result_writer()
test_pytorch_engine()
print("=== 所有测试通过 ===")
