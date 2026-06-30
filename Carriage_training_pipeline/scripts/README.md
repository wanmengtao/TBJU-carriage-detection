# 数据处理脚本

本目录包含数据处理、标注转换、数据扩充等脚本。

---

## 脚本列表

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `convert_labelstudio_tbju.py` | 车号数据转换 | Label Studio JSON | YOLO + OCR 数据集 |
| `convert_labelstudio_new.py` | 异物/车门数据转换 | Label Studio JSON | YOLO txt 标签 |
| `augment_debris.py` | 异物贴图扩充 | 原图 + 素材 | 扩充图 + JSON |
| `augment_video_debris.py` | 视频帧异物扩充 | 视频文件 | 扩充图 + JSON |
| `setup_new_datasets.py` | 同步图片到数据集 | raw_data/ | datasets/ |
| `merge_datasets.py` | 合并数据集 | 4个任务数据集 | 统一数据集 (nc=6) |

---

## 使用方法

### 1. convert_labelstudio_tbju.py — 车号数据转换

将 Label Studio 标注的车号数据转换为 YOLO 检测 + OCR 识别数据集。

```bash
python scripts/convert_labelstudio_tbju.py \
    --dataset_dirs raw_data/eye_level raw_data/side_view \
    --output datasets/output
```

**功能：**
- 解析 Label Studio JSON（坐标百分比 → 像素）
- 文本标准化（TBJU + 7位数字）
- 裁剪车号区域生成 OCR crops
- 划分 train/val/test

**输出：**
- `datasets/output/wagon_number_detection_平视/` — YOLO 检测数据集
- `datasets/output/wagon_number_detection_侧视/`
- `datasets/output/wagon_number_ocr_平视/` — OCR 识别数据集
- `datasets/output/wagon_number_ocr_侧视/`

---

### 2. convert_labelstudio_new.py — 异物/车门数据转换

将 Label Studio 标注的异物/车门数据转换为 YOLO txt 标签。

```bash
python scripts/convert_labelstudio_new.py
```

**功能：**
- 解析 Label Studio JSON
- 坐标转换（百分比 → 归一化）
- 处理带后缀的类别名（如 `track_region_val` → class 0）

**类别映射：**
| 类别名 | class_id |
|--------|----------|
| `track_region` / `carriage_rim_region` / `door_region` | 0 |
| `debris` | 1 |

---

### 3. augment_debris.py — 异物贴图扩充

在原图上贴入异物素材，生成扩充训练数据。

```bash
# 车厢沿异物扩充（只用石头）
python scripts/augment_debris.py \
    --dataset_dir raw_data/carriage_rim_debris \
    --debris_dir raw_data/debris_materials \
    --class_name carriage_rim_region \
    --debris_filter rock

# 轨道异物扩充（所有素材）
python scripts/augment_debris.py \
    --dataset_dir raw_data/track_intrusion \
    --debris_dir raw_data/debris_materials \
    --class_name track_region
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset_dir` | 数据集目录 | 必填 |
| `--debris_dir` | 异物素材目录 | 必填 |
| `--class_name` | 区域类别名 | 必填 |
| `--debris_filter` | 只使用指定素材（如 rock） | 全部 |
| `--augment_test` | 对 test 集也做扩充 | 否 |

**异物基准尺寸：**
| 素材 | 基准尺寸 | 面积 |
|------|----------|------|
| rock | 22px | 484px² |
| coalslag | 30px | 900px² |
| bottle | 28px | 784px² |

**扩充规则：**
- 每张图异物数量：Poisson(λ=2.5) 分布，截断 [1,5]
- 每个异物 ±10% 随机尺寸变化
- 同一异物在同张图中保持一致大小

**输出：**
- `*_aug0.jpg`, `*_aug1.jpg` — 扩充图
- `augmented_debris_train.json` — 扩充标注（Label Studio 格式）

---

### 4. augment_video_debris.py — 视频帧异物扩充

从视频中提取帧并进行异物扩充。

```bash
python scripts/augment_video_debris.py \
    --video_path path/to/video.mp4 \
    --output_dir raw_data/track_intrusion \
    --debris_dir raw_data/debris_materials \
    --class_name track_region
```

---

### 5. setup_new_datasets.py — 同步图片到数据集

将 raw_data/ 中的图片（含扩充图）复制到 datasets/ 目录。

```bash
python scripts/setup_new_datasets.py
```

**功能：**
- 清空目标目录旧图
- 复制原图 + 扩充图到 datasets/
- 保持目录结构一致

**注意：** 运行前确保已执行 `augment_debris.py` 生成扩充图。

---

### 6. merge_datasets.py — 合并数据集

将 4 个任务的数据集合并为统一训练集（nc=6）。

```bash
python scripts/merge_datasets.py
```

**合并类别：**
| 原始类别 | 统一 class_id |
|----------|---------------|
| TBJU_region | 0 |
| track_region | 1 |
| track_debris | 2 |
| carriage_rim_region | 3 |
| carriage_rim_debris | 4 |
| door_region | 5 |

**输出：**
- `datasets/merged_detection/` — 统一数据集

---

## 完整工作流

按照以下顺序执行：

```bash
# 1. 车号数据转换
python scripts/convert_labelstudio_tbju.py --dataset_dirs raw_data/eye_level raw_data/side_view --output datasets/output

# 2. 异物扩充
python scripts/augment_debris.py --dataset_dir raw_data/carriage_rim_debris --debris_dir raw_data/debris_materials --class_name carriage_rim_region --debris_filter rock
python scripts/augment_debris.py --dataset_dir raw_data/track_intrusion --debris_dir raw_data/debris_materials --class_name track_region

# 3. 同步图片
python scripts/setup_new_datasets.py

# 4. 转换标签
python scripts/convert_labelstudio_new.py

# 5. 合并数据集（可选，用于统一模型训练）
python scripts/merge_datasets.py
```

---

## 注意事项

1. **运行顺序**：`augment_debris.py` → `setup_new_datasets.py` → `convert_labelstudio_new.py`
2. **扩充图命名**：`{原图名}_aug{N}.jpg`（如 `0006_S_f0004_aug0.jpg`）
3. **正负样本**：原图是负样本（只有 region），扩充图是正样本（region + debris）
4. **Label Studio 格式**：`file_upload` 字段有 UUID 前缀，脚本会自动处理
