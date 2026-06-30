# TBJU 车号标注数据转换规范

## 1. 背景

使用 Label Studio 人工标注了一批 TBJU 车号区域数据，已导出 JSON 文件。

**当前任务目标：**
- 只做 TBJU 类车号识别
- 最终部署到 ELF2 / RK3588 开发板
- 本阶段将 Label Studio JSON 转换为两个数据集：
  1. **YOLO 检测数据集**：训练车号区域检测模型
  2. **OCR 识别数据集**：训练或校验具体车号文本识别

**注意：** 当前只做 JSON → 数据集转换，不训练模型。等确认 preview 图和 labels.csv 正确后，再进入 YOLO 训练和 RKNN 部署阶段。

---

## 2. 输入约定

### 2.1 Label Studio JSON 文件

- 格式：**JSON**（Label Studio 原生导出格式）
- 示例：`project-3-at-2026-05-18-10-06-5922645e.json`
- 支持任意数量的 JSON 文件输入

**JSON 内部结构：**

```json
[
  {
    "id": 1,
    "annotations": [
      {
        "result": [
          {
            "id": "db8G-Arj9E",
            "type": "rectanglelabels",
            "value": {
              "x": 81.818,
              "y": 35.531,
              "width": 11.819,
              "height": 4.270,
              "rotation": 0,
              "rectanglelabels": ["Carriage Number (Eye Level)"]
            },
            "from_name": "label",
            "to_name": "image"
          },
          {
            "id": "db8G-Arj9E",
            "type": "textarea",
            "value": {
              "x": 81.818,
              "y": 35.531,
              "width": 11.819,
              "height": 4.270,
              "rotation": 0,
              "text": ["TBJU49508821"]
            },
            "from_name": "transcription",
            "to_name": "image"
          }
        ]
      }
    ],
    "data": {
      "image": "/data/upload/7/4e264230-DJI_20260418131852_0003_S_t00000.jpg"
    },
    "file_upload": "4e264230-DJI_20260418131852_0003_S_t00000.jpg"
  }
]
```

**关键点：**
- 框坐标和类别存储在 `type: "rectanglelabels"` 的 result 项中
- 车号文本存储在 `type: "textarea"` 的 result 项中
- **通过 `id` 字段关联框和文本**（相同 id 表示同一个框）
- 图片路径字段为 `data.image`，可能包含完整路径前缀
- `file_upload` 字段也包含文件名

### 2.2 原始图片目录

- 示例：`raw_data/eye_level/train/images`
- 包含所有原始图片
- 支持格式：`.jpg`、`.jpeg`、`.png`（大小写不敏感）

---

## 3. 输出目录

- 示例：`datasets/output`

---

## 4. 输出数据集结构

### 4.1 YOLO 检测数据集

```
wagon_number_detection/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

**类别定义：**

```yaml
names: ['TBJU_region']
nc: 1
```

**YOLO 标签格式（每行一个框）：**

```
class_id x_center y_center width height
```

**坐标转换公式：**

Label Studio 导出的坐标是百分比坐标（0-100），转换为 YOLO 格式（0-1）：

```
x_center = (x + width / 2) / 100
y_center = (y + height / 2) / 100
w = width / 100
h = height / 100
```

**重要：** 即使某个框缺少车号文本，仍然要生成 YOLO 标签（检测模型只需要框）。

### 4.2 OCR 识别数据集

```
wagon_number_ocr/
├── train/
│   └── crops/
├── val/
│   └── crops/
├── labels.csv
├── review_list.csv
└── preview/
```

#### crops 目录

- 命名格式：`{原图文件名}_box{序号}.jpg`（序号从 0 开始）
- **只裁剪有车号文本且通过校验的框**

#### labels.csv 格式

| 列名 | 说明 |
|------|------|
| crop_name | 裁剪图文件名 |
| original_image | 来源原图文件名 |
| text | 标准化后的车号文本 |
| type | 固定为 TBJU_region |
| x1, y1, x2, y2 | 框在原图上的绝对像素坐标 |
| split | train 或 val |

**示例行：**
```csv
crop_name,original_image,text,type,x1,y1,x2,y2,split
DJI_20260418131852_0003_S_t00000_box0.jpg,DJI_20260418131852_0003_S_t00000.jpg,TBJU49508821,TBJU_region,100,150,300,210,train
```

#### review_list.csv

记录需要人工复核的框：

| 列名 | 说明 |
|------|------|
| original_image | 原图文件名 |
| x1, y1, x2, y2 | 框的绝对坐标 |
| text | 原始文本（未标准化） |
| reason | 原因（missing_text / invalid_text / image_not_found） |

#### preview 图片

- 保存在 `wagon_number_ocr/preview/` 下
- 每张原图生成一张预览图
- 框旁边标注车号文本（通过校验的为绿色，未通过的为红色）

---

## 5. 类别处理

**Label Studio 中可能的类别名称：**
- `TBJU_region`
- `Carriage Number (Side Level)`
- `Carriage Number (Eye Level)`

**统一映射：**
- 以上所有类别 → `class_id = 0`（TBJU_region）

---

## 6. 文本处理

### 6.1 文本标准化

从 `textarea` 中提取的文本需要标准化：

1. 去掉空格
2. 去掉横杠
3. 去掉方括号
4. 全部转为大写

**示例：**
- `TBJu0633832` → `TBJU0633832`
- `TBJU 697052 7` → `TBJU6970527`
- `TBJU-4950882` → `TBJU4950882`

### 6.2 文本校验

**正式进入训练集的文本格式：**
```
^TBJU\d{7}$
```

即：TBJU + 7位数字

**以下情况写入 review_list.csv：**
- 缺少文本（missing_text）
- 前缀不是 TBJU（如 TBCU、TCLU）
- 长度不对（不是 TBJU + 7位数字）
- 含有问号或不确定字符

**示例：**
- `TBCU0166455` → review_list.csv（前缀错误）
- `TBJU123` → review_list.csv（长度不够）
- `TBJU?4567890` → review_list.csv（含有不确定字符）

---

## 7. 图片路径匹配

### 7.1 路径来源

JSON 中的图片路径可能来自：
- `data.image`：`/data/upload/7/4e264230-DJI_20260418131852_0003_S_t00000.jpg`
- `file_upload`：`4e264230-DJI_20260418131852_0003_S_t00000.jpg`

### 7.2 本地图片目录

本地图片可能只有：
- `DJI_20260418131852_0003_S_t00000.jpg`

### 7.3 鲁棒匹配策略

按以下顺序尝试匹配：

1. **用 `file_upload` 原名匹配**
2. **去掉 hash 前缀**（去掉第一个 `-` 之前的内容）
3. **按文件名后缀模糊匹配**
4. **如果仍找不到** → 写入 review_list.csv（reason: image_not_found）

---

## 8. 数据集划分

### 8.1 划分规则

- 按图片维度进行 **8:2** 划分（train : val）
- 同一张原图的所有框必须在同一个 split
- 固定随机种子（`random.seed(42)`）保证可重现

### 8.2 同步要求

以下内容必须同步划分：
- 原始图片 → `wagon_number_detection/{split}/images/`
- YOLO 标签 → `wagon_number_detection/{split}/labels/`
- crop 图片 → `wagon_number_ocr/{split}/crops/`
- CSV 记录 → labels.csv 中的 split 字段

---

## 9. 转换步骤

### 9.1 解析 JSON 文件

- 读取所有输入的 JSON 文件
- 对每个 task：
  - 提取 `annotations[0].result` 中的 `rectanglelabels` 和 `textarea`
  - 通过 `id` 配对框和文本
  - 提取图片路径（优先使用 `file_upload`）

### 9.2 匹配本地图片

- 按照 7.3 节的鲁棒匹配策略查找本地图片
- 找不到的写入 review_list.csv

### 9.3 处理文本

- 按 6.1 节标准化文本
- 按 6.2 节校验文本格式
- 不通过校验的写入 review_list.csv

### 9.4 生成 YOLO 标签

- 对每个有效框（有文本且通过校验）
- 按公式转换坐标
- 写入对应的 txt 文件

### 9.5 生成 OCR 数据集

- 裁剪有效框的区域
- 保存到 crops 目录
- 写入 labels.csv

### 9.6 生成 preview 图

- 在原图上画框和文本
- 保存到 preview 目录

### 9.7 划分数据集

- 按 8:2 比例划分
- 同步复制图片和标签

### 9.8 输出统计

生成 `summary.txt`（见第 11 节）

---

## 10. 脚本执行方式

**脚本文件名：** `convert_labelstudio_tbju.py`

**执行前请确认以下路径：**

1. JSON 文件路径
2. 原始图片目录路径
3. 输出目录路径

**命令格式：**

```bash
python convert_labelstudio_tbju.py \
  --json path/to/project-3-at-2026-05-18-10-06-5922645e.json \
  --dataset_dirs raw_data/eye_level raw_data/side_view \
  --output datasets/output \
  --seed 42
```

**参数说明：**

| 参数 | 说明 | 必填 |
|------|------|------|
| --json | 一个或多个 Label Studio 导出的 JSON 文件 | 是 |
| --images | 原始图片所在目录 | 是 |
| --output | 输出根目录 | 是 |
| --seed | 随机种子（默认 42） | 否 |

---

## 11. 输出统计（summary.txt）

脚本运行结束后打印并保存 `summary.txt`：

| 统计项 | 说明 |
|--------|------|
| total_tasks | 总 task 数 |
| matched_images | 成功匹配图片数量 |
| unmatched_images | 找不到图片数量 |
| total_boxes | 总标注框数量 |
| valid_tbju_boxes | 有效 TBJU 标注数量 |
| review_count | review 数量 |
| train_images | train 图片数量 |
| val_images | val 图片数量 |
| train_crops | train crop 数量 |
| val_crops | val crop 数量 |

---

## 12. 注意事项

### 缺少车号文本的处理

| 场景 | 处理方式 |
|------|---------|
| YOLO 数据集 | 包含这些框（检测模型只需要框） |
| OCR 数据集 | 跳过这些框 |
| review_list.csv | 记录，reason: missing_text |

### 无效文本的处理

| 场景 | 处理方式 |
|------|---------|
| 前缀错误（TBCU等） | review_list.csv，reason: invalid_text |
| 长度不对 | review_list.csv，reason: invalid_text |
| 含不确定字符 | review_list.csv，reason: invalid_text |

### 图片找不到的处理

| 场景 | 处理方式 |
|------|---------|
| 本地图片不存在 | review_list.csv，reason: image_not_found |

---

## 13. 依赖库

- 标准库：`json`, `csv`, `os`, `re`, `random`, `argparse`, `pathlib`, `collections`
- 第三方库：`Pillow`（`pip install Pillow`）

---

## 14. 最终交付物

生成 `convert_labelstudio_tbju.py` 脚本，满足以上所有要求，包含错误处理和完整日志输出。

---

## 15. 数据集检查工具

### 15.1 检查数据集完整性

```python
import os

def check_dataset(images_dir, labels_dir):
    """检查图片和标签文件是否匹配"""
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def clean_name(filename):
        name = os.path.splitext(filename)[0]
        if '-' in name and len(name.split('-')[0]) == 8:
            return '-'.join(name.split('-')[1:])
        return name

    label_names = set(clean_name(f) for f in label_files)
    image_names = set(clean_name(f) for f in image_files)

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print(f'标签文件: {len(label_files)} 个')
    print(f'图片文件: {len(image_files)} 个')

    if not missing_labels and not missing_images:
        print('[OK] 完全匹配！')
    else:
        if missing_labels:
            print(f'[X] {len(missing_labels)} 张图片没有标签')
        if missing_images:
            print(f'[X] {len(missing_images)} 个标签没有图片')

    return len(missing_labels) == 0 and len(missing_images) == 0
```

### 15.2 批量重命名去掉 UUID 前缀

```python
import os

def rename_labels(labels_dir):
    """去掉标签文件名中的UUID前缀"""
    files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    renamed_count = 0

    for f in files:
        name = os.path.splitext(f)[0]
        if '-' in name and len(name.split('-')[0]) == 8:
            new_name = '-'.join(name.split('-')[1:])
        else:
            new_name = name

        old_path = os.path.join(labels_dir, f)
        new_path = os.path.join(labels_dir, new_name + '.txt')

        if old_path != new_path:
            os.rename(old_path, new_path)
            renamed_count += 1

    print(f'重命名完成: {renamed_count} 个文件')
    return renamed_count
```

---

## 16. 数据集处理流程检查清单

### 第一步：运行转换脚本
- [ ] 确认 JSON 文件路径
- [ ] 确认原始图片目录路径
- [ ] 确认输出目录路径
- [ ] 运行 `convert_labelstudio_tbju.py`

### 第二步：检查输出
- [ ] 查看 summary.txt 统计信息
- [ ] 检查 preview 目录中的预览图
- [ ] 检查 labels.csv 内容
- [ ] 检查 review_list.csv 中的记录

### 第三步：确认数据质量
- [ ] 抽样检查几个 crop 图片
- [ ] 确认文本标准化正确
- [ ] 确认坐标框位置正确

### 第四步：进入训练阶段
- [ ] 确认数据集无误后，进入 YOLO 训练
