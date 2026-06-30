# ELF2 / RK3588 车号识别：ONNX → RKNN → 板端部署路线（推荐不用 TrOCR）

> 目标：把训练好的 TBJU 车号识别系统部署到 ELF2（RK3588）开发板上。  
> 推荐路线：**YOLOv8n/YOLOv5s 检测车号区域 + PP-OCR Rec / CRNN 类轻量 OCR 识别车号文本**。  
> 不推荐优先使用 TrOCR 做板端部署，因为 TrOCR 属于 Transformer OCR，ONNX 导出、RKNN 转换、板端推理复杂度和不确定性都明显高于 PP-OCR Rec / CRNN / LPRNet 这类嵌入式友好模型。

---

## 0. 总体路线

最终系统建议分成两个模型：

```text
整张无人机图片
    ↓
模型 1：YOLO 检测 TBJU 车号区域
    ↓
裁剪 TBJU 车号区域 crop
    ↓
模型 2：PP-OCR Rec / CRNN 识别 crop 里的字符
    ↓
输出 TBJU6970527 / TBJU4950882 等具体车号
```

部署到 ELF2 时，两个模型都要尽量转换成 RKNN：

```text
YOLO best.pt
    ↓ export
YOLO best.onnx
    ↓ RKNN-Toolkit2 2.3.2
YOLO best.rknn
    ↓ ELF2 / RK3588

OCR 模型
    ↓ export
OCR rec.onnx
    ↓ RKNN-Toolkit2 2.3.2
OCR rec.rknn
    ↓ ELF2 / RK3588
```

---

## 1. 为什么不用 TrOCR，改用 PP-OCR Rec / CRNN

TrOCR 在电脑端训练可能效果很好，但它是 Transformer 架构，通常包含较复杂的 attention、decoder、token 生成逻辑。部署到 RK3588 NPU 时，可能会遇到：

```text
ONNX 导出复杂
ONNX 动态 shape 难处理
RKNN 不支持某些算子
后处理复杂
板端速度不稳定
```

所以建议改为更适合嵌入式部署的文字识别模型：

```text
优先推荐：PP-OCR Rec
备选方案：CRNN
备选方案：LPRNet 改造为 TBJU 字符识别
```

其中 PP-OCR Rec 的优势是 Rockchip Model Zoo / ELF2 例程里已经有 PPOCR 字符识别相关流程，和 RKNN 部署链路更接近。CRNN/LPRNet 的优势是结构简单，通常比 TrOCR 更容易导出 ONNX 和转 RKNN。

---

## 2. 数据准备阶段

你们用 Label Studio 标完之后，导出 JSON。不要直接用 Label Studio 的 YOLO 导出作为最终数据源，因为 YOLO txt 只保留框，不保留车号文本。

推荐保留：

```text
dataset_root/
├── images/
│   ├── xxx.jpg
│   └── ...
└── labels/
    └── label_studio_export.json
```

然后让转换脚本从 JSON 生成两套数据。

---

## 3. 从 Label Studio JSON 生成两套数据

### 3.1 YOLO 检测数据集

输出结构：

```text
wagon_number_detection/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
├── test/images/
├── test/labels/
└── dataset.yaml
```

类别只保留：

```text
0 = TBJU_region
```

YOLO 标签格式：

```text
class_id x_center y_center width height
```

例如：

```text
0 0.512345 0.236111 0.082500 0.022222
```

这个数据集训练出来的模型只负责回答：

```text
TBJU 车号区域在哪里？
```

它不负责识别具体文本。

---

### 3.2 OCR 识别数据集

输出结构：

```text
wagon_number_ocr/
├── train/crops/
├── val/crops/
├── test/crops/
├── labels_train.txt 或 labels_train.csv
├── labels_val.txt 或 labels_val.csv
├── labels_test.txt 或 labels_test.csv
└── preview/
```

每个 crop 是从原图中裁剪出的 TBJU 车号区域。

OCR 标签示例：

```text
crop_000001.jpg    TBJU6970527
crop_000002.jpg    TBJU4950882
```

这个数据集训练出来的模型负责回答：

```text
这个车号 crop 里写的具体文本是什么？
```

---

## 4. YOLO 检测模型训练与导出

### 4.1 训练环境

建议在 Windows 或服务器上训练，不在 ELF2 开发板上训练。

需要：

```text
Python
PyTorch
ultralytics
opencv-python
```

训练命令示例：

```bash
cd YOLO_train
python train_yolo.py --use_config
```

如果不用自写脚本，也可以直接用 ultralytics：

```bash
yolo detect train model=yolov8n.pt data="datasets/output/wagon_number_detection_平视/dataset.yaml" imgsz=640 epochs=100 batch=16
```

建议第一版模型选择：

```text
yolov8n 或 yolov5s
```

不要一开始用太大的模型。小模型更容易部署到 RK3588，速度更稳。

---

### 4.2 YOLO 训练后你应该得到什么

训练完成后，重点看：

```text
best.pt
results.png
PR_curve.png
confusion_matrix.png
验证集预测图
```

最重要的模型文件：

```text
runs/detect/train/weights/best.pt
```

效果目标：

```text
mAP50 尽量高
Precision 高
Recall 高
预测框只框 TBJU 车号，不框中国铁路、2NUA、公司名
```

---

### 4.3 导出 YOLO ONNX

不要优先用 `--export_rknn` 自动导出 RKNN。建议先导出 ONNX，再在 RKNN-Toolkit2 环境里手动转 RKNN。

示例：

```bash
yolo export model="runs/detect/train/weights/best.pt" format=onnx opset=12 imgsz=640 simplify=True
```

如果 opset=12 不成功，可以试：

```bash
yolo export model="runs/detect/train/weights/best.pt" format=onnx opset=15 imgsz=640 simplify=True
```

导出后得到：

```text
best.onnx
```

注意：如果导出后出现 `best.onnx.data`，说明权重被拆分了，后续 RKNN 转换容易出问题，建议调整 PyTorch/ONNX 导出方式，确保最终是单个 `best.onnx`。

---

## 5. OCR 识别模型训练路线：推荐 PP-OCR Rec / CRNN

### 5.1 推荐路线 A：PP-OCR Rec

这是更贴近 Rockchip Model Zoo 的方案。

流程：

```text
OCR crops + labels
    ↓
PaddleOCR 识别模型微调
    ↓
导出 Paddle inference model
    ↓
paddle2onnx 转 ONNX
    ↓
RKNN-Toolkit2 转 RKNN
    ↓
ELF2 板端推理
```

### 5.2 OCR 字符集

你们只识别 TBJU 车号，字符集可以很小：

```text
T B J U 0 1 2 3 4 5 6 7 8 9
```

如果后续加入 TBCU、TCLU，则扩展为：

```text
A-Z + 0-9
```

但第一版建议先只做 TBJU。

---

### 5.3 OCR 训练数据格式

PP-OCR Rec 常见格式是：

```text
图片路径<TAB>文本标签
```

例如：

```text
train/crops/crop_000001.jpg    TBJU6970527
train/crops/crop_000002.jpg    TBJU4950882
```

注意文本不要加空格：

```text
正确：TBJU6970527
错误：TBJU 697052 7
```

---

### 5.4 OCR 训练后你应该看到什么

OCR 单独测试时，输入 crop，输出字符串：

```text
输入：crop_000001.jpg
输出：TBJU6970527
```

重点看：

```text
字符串准确率
预测文本 vs 真实文本
错误样本列表
```

不要只看 loss。车号识别里只要错一个字符，整条车号就算错。

---

### 5.5 导出 OCR ONNX

如果使用 PaddleOCR / PP-OCR Rec，大致流程是：

```text
训练得到 Paddle 模型
    ↓
导出 inference model
    ↓
使用 paddle2onnx 转为 rec.onnx
```

示例逻辑：

```bash
paddle2onnx --model_dir ./inference_rec_model \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file rec_tbju.onnx \
            --opset_version 12
```

具体文件名以你们训练输出为准。

---

## 6. 在 Ubuntu 虚拟机中做 RKNN 转换

你们已经配置过：

```text
RKNN-Toolkit2-2.3.2
rknn_model_zoo-2.3.2
```

ELF2 / RK3588 建议使用匹配的版本：

```text
RKNN-Toolkit2 2.3.2
RKNN-Toolkit-Lite2 2.3.2
target_platform = rk3588
```

---

### 6.1 激活 RKNN 环境

在 Ubuntu 虚拟机中：

```bash
conda activate RKNN-Toolkit2-2.3.2
```

检查：

```bash
python -c "from rknn.api import RKNN; print('RKNN Toolkit2 OK')"
```

---

### 6.2 准备量化校准数据

量化时需要一个 `dataset.txt`，里面每行是一张校准图片路径。

YOLO 校准数据建议用你们真实场景的原图：

```text
/home/elf/tbju_calib/images/img001.jpg
/home/elf/tbju_calib/images/img002.jpg
...
```

OCR 校准数据建议用车号 crop：

```text
/home/elf/tbju_calib/ocr_crops/crop001.jpg
/home/elf/tbju_calib/ocr_crops/crop002.jpg
...
```

校准数据必须尽量接近实际部署场景，否则量化后可能出现框乱、置信度异常、识别失效。

---

## 7. YOLO ONNX 转 RKNN

建议基于 RKNN Model Zoo 的 YOLOv8 转换脚本改，而不是自己从零写。

大致流程：

```bash
cd ~/rknn_model_zoo-2.3.2/examples/yolov8/python
```

把你训练得到的：

```text
best.onnx
```

复制到该目录或指定路径。

修改转换脚本中的模型路径、输出路径、target：

```text
MODEL_PATH = "/home/elf/models/tbju_yolov8n.onnx"
RKNN_MODEL = "/home/elf/models/tbju_yolov8n.rknn"
DATASET_PATH = "/home/elf/models/yolo_dataset.txt"
target_platform = "rk3588"
```

然后运行：

```bash
python convert.py
```

如果你们脚本不是 `convert.py`，以 Model Zoo 对应 YOLOv8 例程里的转换脚本为准。

转换时必须注意：

```text
mean_values / std_values 要与训练和推理预处理一致
输入尺寸要与导出 ONNX 时一致，例如 640x640
opset 不要太高，建议 12 或 15
target_platform 设置 rk3588
```

成功后得到：

```text
tbju_yolov8n.rknn
```

---

## 8. OCR ONNX 转 RKNN

如果使用 PP-OCR Rec，建议先跑通官方 Model Zoo 的 PPOCR 字符识别例程，再替换成自己的 rec_tbju.onnx。

大致路径：

```bash
cd ~/rknn_model_zoo-2.3.2/examples/ppocr/python
```

或者你们 Model Zoo 中对应的 PPOCR Rec 目录。

修改：

```text
MODEL_PATH = "/home/elf/models/rec_tbju.onnx"
RKNN_MODEL = "/home/elf/models/rec_tbju.rknn"
DATASET_PATH = "/home/elf/models/ocr_dataset.txt"
target_platform = "rk3588"
```

然后运行转换脚本：

```bash
python convert.py
```

如果转换失败，优先检查：

```text
ONNX 是否有不支持算子
输入 shape 是否固定
是否包含动态 shape
是否 opset 过高
是否预处理和模型输入不一致
```

如果 PP-OCR Rec 转换太复杂，备选方案是训练更简单的 CRNN，然后导出固定输入尺寸 ONNX，再转 RKNN。

---

## 9. 在虚拟机中测试 RKNN 模型

转换成功后，先不要马上上板。先在虚拟机里用 RKNN Toolkit 做 simulator 测试：

```text
YOLO RKNN 输入测试图，输出框是否正常
OCR RKNN 输入 crop，输出文本 logits 是否正常
```

注意：虚拟机 simulator 测试通过，不代表板端一定完全一致，但至少能排除模型文件本身明显错误。

---

## 10. 把模型和代码传到 ELF2 开发板

需要传到板端的内容：

```text
tbju_yolov8n.rknn
rec_tbju.rknn
test_images/
inference_tbju.py
label_dict.txt 或 character_dict.txt
```

可以用 U 盘、scp、共享目录等方式。

如果板子 IP 是 `192.168.1.100`，可以：

```bash
scp tbju_yolov8n.rknn elf@192.168.1.100:/home/elf/tbju_demo/
scp rec_tbju.rknn elf@192.168.1.100:/home/elf/tbju_demo/
scp inference_tbju.py elf@192.168.1.100:/home/elf/tbju_demo/
```

---

## 11. ELF2 板端环境检查

在 ELF2 开发板上执行：

```bash
pip list | grep rknn-toolkit-lite2
```

期望看到：

```text
rknn-toolkit-lite2 2.3.2
```

如果没有，需要安装 ELF2 对应的 `rknn_toolkit_lite2-2.3.2-...aarch64.whl`。

安装示例：

```bash
sudo apt-get update
sudo apt-get install python3-pip
pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

---

## 12. 板端 Python 推理代码结构

板端最终推理代码建议叫：

```text
inference_tbju.py
```

核心步骤：

```text
1. 读取整张图片
2. 做 YOLO 预处理：resize / letterbox / normalize
3. 加载 tbju_yolov8n.rknn
4. YOLO RKNN 推理
5. YOLO 后处理：decode + NMS + 坐标还原
6. 根据检测框裁剪 TBJU 区域
7. 做 OCR 预处理：resize 到 OCR 输入尺寸
8. 加载 rec_tbju.rknn
9. OCR RKNN 推理
10. CTC decode 或 OCR decode
11. 输出最终车号文本
12. 保存可视化结果图
```

最终你希望看到：

```text
input: DJI_xxx.jpg
detected bbox: [x1, y1, x2, y2]
recognized text: TBJU6970527
confidence: 0.xx
```

并生成：

```text
result.jpg
result.csv
```

---

## 13. 板端运行命令

在 ELF2 上：

```bash
cd /home/elf/tbju_demo
python3 inference_tbju.py --image test_images/test001.jpg
```

期望输出：

```text
YOLO detect: 1 TBJU region
OCR result: TBJU6970527
Saved result to result/test001_result.jpg
```

批量测试：

```bash
python3 inference_tbju.py --image_dir test_images --save_dir results
```

---

## 14. 端到端测试指标

部署前后都要看三个层面的结果。

### 14.1 YOLO 单独测试

```text
输入整张图
输出 TBJU bbox
```

看：

```text
是否漏检
是否误检
框是否贴合
是否框到中国铁路、2NUA、公司名
```

### 14.2 OCR 单独测试

```text
输入人工 crop
输出 TBJU 文本
```

看：

```text
字符串准确率
错误字符位置
容易混淆的数字
```

### 14.3 端到端测试

```text
整张图 → YOLO → crop → OCR → 车号文本
```

这是最终指标。

一条结果只有在下面两个条件都满足时才算正确：

```text
YOLO 框到了正确车号区域
OCR 文本完全正确
```

---

## 15. 常见问题与排查

### 15.1 YOLO 在 PC 上正常，RKNN 后出现 null 框

重点检查：

```text
量化校准数据是否来自真实场景
mean/std 是否和训练一致
RGB/BGR 是否一致
letterbox 是否一致
后处理 decode 是否适配 RKNN 输出
NMS 阈值是否合理
RKNN-Toolkit2 和 Lite2 版本是否一致
```

---

### 15.2 OCR 在 PC 上正常，RKNN 后识别乱码

重点检查：

```text
OCR 输入 crop 是否 resize 到固定尺寸
是否保持宽高比例
字符字典顺序是否和训练一致
CTC decode 是否一致
ONNX 输出维度是否和 RKNN 输出维度一致
是否有量化精度损失
```

---

### 15.3 转 RKNN 失败

重点检查：

```text
ONNX opset 是否过高
是否存在动态 shape
是否存在 RKNN 不支持算子
是否有 onnx.data 外部权重文件
是否模型过大或结构太复杂
```

建议：

```text
YOLO 用 yolov8n/yolov5s
OCR 用 PP-OCR Rec / CRNN
避免 TrOCR 作为第一版部署模型
```

---

## 16. 最推荐的执行顺序

```text
1. 保留 Label Studio JSON + images
2. 转换为 YOLO 检测数据集和 OCR crop 数据集
3. 训练 YOLOv8n 检测模型
4. 导出 YOLO best.onnx
5. 用 RKNN-Toolkit2 2.3.2 转 YOLO best.rknn
6. 在虚拟机 simulator 测试 YOLO RKNN
7. 在 ELF2 上测试 YOLO RKNN
8. 使用 PP-OCR Rec / CRNN 训练 OCR 模型
9. 导出 OCR rec.onnx
10. 用 RKNN-Toolkit2 2.3.2 转 OCR rec.rknn
11. 在虚拟机 simulator 测试 OCR RKNN
12. 在 ELF2 上测试 OCR RKNN
13. 合并成端到端 inference_tbju.py
14. 输入整张图，输出车号框和 TBJU 文本
```

---

## 17. 项目验收时应该展示什么

最终展示不应该只是 loss 或 accuracy，而应该包括：

```text
1. 原图
2. YOLO 框出的 TBJU 区域
3. 裁剪出的车号 crop
4. OCR 输出文本
5. 真实文本
6. 是否完全正确
7. 板端运行截图
8. 推理耗时
```

示例：

```text
image: DJI_20260418_xxx.jpg
bbox: [1432, 221, 1541, 247]
ocr: TBJU6970527
gt: TBJU6970527
status: correct
yolo_time: 18 ms
ocr_time: 6 ms
device: ELF2 / RK3588
```

---

## 18. 一句话结论

如果目标是“能部署到 ELF2 / RK3588”，第一版不要把 TrOCR 作为主路线。建议使用：

```text
YOLOv8n 检测 TBJU 车号区域
+
PP-OCR Rec 或 CRNN 识别 TBJU 字符串
+
RKNN-Toolkit2 2.3.2 转 RKNN
+
RKNN-Toolkit-Lite2 2.3.2 在 ELF2 上推理
```
