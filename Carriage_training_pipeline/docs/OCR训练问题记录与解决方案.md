# OCR 训练问题记录与解决方案

本文档记录了在训练 PP-OCR Rec 车号识别模型过程中遇到的错误及对应的解决方法。

---

## 1. 混合精度训练兼容性错误

### 错误现象

```text
TypeError: __init__() got an unexpected keyword argument 'device_type'
```

出现在使用 `autocast(device_type="cuda")` 时。

### 原因分析

- 脚本开头使用了 `try-except` 兼容导入，但在 PyTorch 2.2.0 环境下，`torch.cuda.amp.autocast` 的构造函数不接受 `device_type` 参数。
- 调用时传递了 `device_type="cuda"` 导致类型错误。

### 解决方法

1. 删除或注释掉兼容导入块：

```python
# try:
#     from torch.amp import autocast, GradScaler
# except ImportError:
#     from torch.cuda.amp import autocast, GradScaler
```

2. 将所有 `autocast` 调用替换为 `torch.amp.autocast`：

```python
with torch.amp.autocast("cuda"):
```

3. 将 `GradScaler` 调用替换为 `torch.cuda.amp.GradScaler`：

```python
scaler = torch.cuda.amp.GradScaler()
```

> **注意：** `GradScaler` 仍位于 `torch.cuda.amp` 模块，不要改为 `torch.amp.GradScaler`。

---

## 2. CTC Loss 不支持半精度（fp16）

### 错误现象

```text
RuntimeError: "ctc_loss_cuda" not implemented for 'Half'
```

### 原因分析

开启混合精度训练时，CTC Loss 接收了 `float16` 类型的输入，但 CTC 底层 CUDA 算子不支持半精度计算。

### 解决方法

在计算 CTC Loss 时将输入强制转换为 `float32`：

```python
loss = ctc_loss_fn(log_probs_t.float(), labels, input_lengths, label_lengths)
```

---

## 3. Loss 始终为 0 与解码索引越界

### 错误现象

- 训练时 Loss 显示 `0.0000`，验证准确率为 `0.0000`。
- 验证阶段出现索引越界：

```text
IndexError: list index out of range
```

出现在 `ctc_decode` 函数中的 `chars_list.append(chars[idx])`。

### 原因分析

`ctc_decode` 函数错误处理了 CTC 输出索引：

- 索引 `0` 为 blank 但未跳过。
- 有效字符索引范围是 `1 ~ len(chars)`，而 `chars` 列表索引从 `0` 开始，导致越界。
- Loss 为 0 可能也与解码时的错误处理有关，造成验证中断或模型输出异常。

### 解决方法

重写 `ctc_decode` 函数（位于 `ocr_model.py`），修正索引映射：

```python
def ctc_decode(log_probs, chars, blank=0):
    """
    CTC 贪心解码
    log_probs: Tensor shape (batch, T, num_classes)
    chars: 字符列表，不含 blank
    blank: blank 索引，默认为 0
    返回: 解码后的字符串列表
    """
    indices = log_probs.argmax(dim=-1)  # (batch, T)
    batch_results = []
    for b in range(indices.size(0)):
        seq = []
        prev = None
        for t in range(indices.size(1)):
            idx = indices[b, t].item()
            if idx != blank and idx != prev:
                if 1 <= idx <= len(chars):
                    seq.append(chars[idx - 1])
            prev = idx
        batch_results.append(''.join(seq))
    return batch_results
```

---

## 4. 环境依赖问题：NumPy 版本冲突

### 错误现象

```text
RuntimeError: Numpy is not available
```

并提示：

```text
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash.
```

### 原因分析

PyTorch 2.2.0 编译时使用 NumPy 1.x API，而环境中安装了 NumPy 2.0.2，两者不兼容。

### 解决方法

降级 NumPy 到 1.x 版本：

```bash
pip install "numpy<2"
```

---

## 5. 训练脚本文件找不到

### 错误现象

```text
python: can't open file 'train_yolo.py': [Errno 2] No such file or directory
```

### 原因分析

当前工作目录下不存在 `train_yolo.py`，需要先切换到脚本所在目录。

### 解决方法

```bash
cd YOLO_train
python train_yolo.py --use_config --export_rknn
```

---

## 总结

上述问题均已在代码层面修复，最终训练流程可正常执行并导出 ONNX 模型。主要修改点：

- 统一使用 `torch.amp.autocast` 和 `torch.cuda.amp.GradScaler`
- CTC Loss 强制使用 `float32`
- 修正 CTC 解码函数索引逻辑
- 环境依赖降级 NumPy 至 1.x

修改后的脚本能顺利完成训练、评估和 ONNX 导出，产出 `ppocr_rec_tbju.onnx` 用于后续 RKNN 转换。
