# 测试目录

本目录包含单元测试和冒烟测试脚本。

---

## 测试文件

| 文件 | 测试项 | 说明 |
|------|--------|------|
| `test_core.py` | 67 项 | pytest 单元测试 |
| `smoke_core.py` | 4 项 | 冒烟测试（快速验证） |
| `__init__.py` | — | 包初始化 |

---

## 运行测试

### pytest 单元测试（67 项）

```bash
# 运行所有测试
python tests/test_core.py

# 使用 pytest 运行
pytest tests/test_core.py -v

# 运行特定测试
pytest tests/test_core.py::TestDecodeYolov8 -v
```

**测试覆盖：**
| 模块 | 测试内容 |
|------|----------|
| `decode_yolov8()` | YOLO 输出解码 |
| `validate_debris_region()` | 区域验证 |
| `TemporalConsistencyFilter` | 时序一致性滤波 |
| `load_classes()` | 类别加载和校验 |
| `expand_box()` | 框扩展 |
| `preprocess()` | 图像预处理 |
| `draw_detections()` | 绘制检测结果 |
| `safe_csv_cell()` | CSV 安全写入 |
| CTC 解码 | OCR 解码逻辑 |

### 冒烟测试（4 项）

```bash
python run_smoke_test.py
```

**测试内容：**
1. `test_load_functions()` — 配置加载（6 类 + 15 字符）
2. `test_preprocess()` — 预处理形状验证
3. `test_result_writer()` — CSV 写入验证
4. `test_pytorch_engine()` — 完整推理（模型文件缺失时自动跳过）

---

## 测试结果

测试通过输出：

```
============================= test session starts =============================
platform win32 -- Python 3.10.x, pytest-7.x.x
collected 67 items

tests/test_core.py::TestDecodeYolov8::test_basic PASSED                  [  1%]
tests/test_core.py::TestDecodeYolov8::test_empty PASSED                  [  2%]
...
tests/test_core.py::TestSafeCsvCell::test_injection PASSED               [100%]

============================= 67 passed in 0.45s ==============================
```

---

## 注意事项

1. **模型文件**：`test_pytorch_engine()` 需要模型文件，缺失时自动跳过
2. **运行环境**：Windows PC 和 RK3588 均可运行
3. **依赖**：需要安装 `pytest`（`pip install pytest`）
