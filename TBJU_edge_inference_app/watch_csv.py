"""
CSV 文件监控 + TXT 摘要生成器
==============================
监控 output 目录下新出现的 CSV 文件，自动生成可读的 TXT 摘要。

启动: python watch_csv.py
输出: 与 CSV 同目录，文件名相同但扩展名为 .txt

无需额外依赖，纯标准库实现。
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# 配置
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
POLL_INTERVAL = 2  # 秒
PROCESSED_FILE = Path(__file__).resolve().parent / ".csv_processed"


def load_processed():
    """加载已处理文件列表"""
    if PROCESSED_FILE.exists():
        return set(PROCESSED_FILE.read_text(encoding="utf-8").splitlines())
    return set()


def save_processed(processed):
    """保存已处理文件列表"""
    PROCESSED_FILE.write_text("\n".join(sorted(processed)), encoding="utf-8")


def format_number(val):
    """格式化数字"""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return f"{f:.1f}"
    except (ValueError, TypeError):
        return str(val) if val else "-"


def generate_result_txt(csv_path, txt_path):
    """从 result.csv 生成可读 TXT 摘要"""
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"  读取失败: {e}")
        return False

    if not rows:
        return False

    lines = []
    lines.append("=" * 60)
    lines.append("  TBJU 检测结果摘要")
    lines.append("=" * 60)
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  源文件: {csv_path.name}")
    lines.append(f"  总记录数: {len(rows)}")
    lines.append("")

    # 来源和帧范围
    sources = set()
    frames = []
    for r in rows:
        if r.get("source"):
            sources.add(r["source"])
        if r.get("frame"):
            try:
                frames.append(int(r["frame"]))
            except ValueError:
                pass

    lines.append("-" * 60)
    lines.append("  基本信息")
    lines.append("-" * 60)
    lines.append(f"  数据来源: {', '.join(sources) if sources else '-'}")
    if frames:
        lines.append(f"  帧范围: {min(frames)} ~ {max(frames)}")
    lines.append("")

    # 类别统计
    class_counts = {}
    ocr_texts = []
    confidences = []
    yolo_times = []
    ocr_times = []

    for r in rows:
        cls_name = r.get("class_name", "unknown")
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        if r.get("ocr_text"):
            ocr_texts.append(r["ocr_text"])

        try:
            confidences.append(float(r.get("det_conf", 0)))
        except ValueError:
            pass

        try:
            yolo_times.append(float(r.get("yolo_ms", 0)))
        except ValueError:
            pass

        try:
            ocr_times.append(float(r.get("ocr_ms", 0)))
        except ValueError:
            pass

    lines.append("-" * 60)
    lines.append("  检测类别统计")
    lines.append("-" * 60)
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / len(rows) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {cls_name:<25s} {count:>4d} 次  {bar} {pct:.1f}%")
    lines.append("")

    # 置信度统计
    if confidences:
        lines.append("-" * 60)
        lines.append("  置信度统计")
        lines.append("-" * 60)
        lines.append(f"  平均值: {sum(confidences)/len(confidences):.3f}")
        lines.append(f"  最高值: {max(confidences):.3f}")
        lines.append(f"  最低值: {min(confidences):.3f}")
        lines.append("")

    # OCR 识别结果
    if ocr_texts:
        lines.append("-" * 60)
        lines.append("  OCR 识别结果")
        lines.append("-" * 60)
        unique_ocr = list(dict.fromkeys(ocr_texts))  # 保持顺序去重
        for text in unique_ocr:
            count = ocr_texts.count(text)
            lines.append(f"  {text:<20s} 出现 {count} 次")
        lines.append("")

    # 性能统计
    if yolo_times:
        valid_yolo = [t for t in yolo_times if t > 0]
        if valid_yolo:
            lines.append("-" * 60)
            lines.append("  性能统计")
            lines.append("-" * 60)
            lines.append(f"  YOLO 平均耗时: {sum(valid_yolo)/len(valid_yolo):.1f} ms")
            lines.append(f"  YOLO 最快: {min(valid_yolo):.1f} ms")
            lines.append(f"  YOLO 最慢: {max(valid_yolo):.1f} ms")

    if ocr_times:
        valid_ocr = [t for t in ocr_times if t > 0]
        if valid_ocr:
            lines.append(f"  OCR 平均耗时: {sum(valid_ocr)/len(valid_ocr):.1f} ms")

    lines.append("")
    lines.append("=" * 60)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def generate_metrics_txt(csv_path, txt_path):
    """从 metrics CSV 生成可读 TXT 摘要"""
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"  读取失败: {e}")
        return False

    if not rows:
        return False

    lines = []
    lines.append("=" * 60)
    lines.append("  TBJU 系统指标摘要")
    lines.append("=" * 60)
    lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  源文件: {csv_path.name}")
    lines.append(f"  采样点数: {len(rows)}")
    lines.append("")

    # 提取数值列
    def get_floats(key):
        vals = []
        for r in rows:
            try:
                v = float(r.get(key, 0))
                if v > 0:
                    vals.append(v)
            except (ValueError, TypeError):
                pass
        return vals

    metrics = [
        ("cpu_percent", "CPU 使用率", "%"),
        ("memory_percent", "内存使用率", "%"),
        ("memory_used_mb", "内存使用量", "MB"),
        ("temp_max_c", "最高温度", "°C"),
        ("npu_load_percent", "NPU 负载", "%"),
    ]

    lines.append("-" * 60)
    lines.append("  指标统计")
    lines.append("-" * 60)

    for key, label, unit in metrics:
        vals = get_floats(key)
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"  {label}:")
            lines.append(f"    平均: {avg:.1f}{unit}  最高: {max(vals):.1f}{unit}  最低: {min(vals):.1f}{unit}")

    # 时间范围
    timestamps = [r.get("timestamp", "") for r in rows if r.get("timestamp")]
    if timestamps:
        lines.append("")
        lines.append(f"  时间范围: {timestamps[0]} ~ {timestamps[-1]}")

    lines.append("")
    lines.append("=" * 60)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return True


def process_csv(csv_path):
    """处理单个 CSV 文件"""
    csv_path = Path(csv_path)
    txt_path = csv_path.with_suffix(".txt")

    # 跳过已存在 TXT 的
    if txt_path.exists():
        return True

    # 跳过太小的文件（只有 header 或空文件）
    if csv_path.stat().st_size < 20:
        return True

    print(f"  生成摘要: {txt_path.name}")

    # 根据文件名判断类型
    if "metrics" in csv_path.name:
        return generate_metrics_txt(csv_path, txt_path)
    else:
        return generate_result_txt(csv_path, txt_path)


def scan_once(processed):
    """扫描一次 output 目录"""
    if not OUTPUT_DIR.exists():
        return processed

    # 清理已不存在的文件记录，防止无限增长
    cleaned = set()
    for key in processed:
        if Path(key).exists():
            cleaned.add(key)
    if len(cleaned) < len(processed):
        processed = cleaned

    new_files = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in files:
            if not fname.endswith(".csv"):
                continue
            csv_path = Path(root) / fname
            key = str(csv_path.resolve())
            if key not in processed:
                new_files.append(csv_path)

    if not new_files:
        return processed

    print(f"发现 {len(new_files)} 个新 CSV 文件")
    for csv_path in sorted(new_files):
        key = str(csv_path.resolve())
        if process_csv(csv_path):
            processed.add(key)

    save_processed(processed)
    return processed


def main():
    print("=" * 60)
    print("  TBJU CSV 文件监控器")
    print("=" * 60)
    print(f"  监控目录: {OUTPUT_DIR.resolve()}")
    print(f"  轮询间隔: {POLL_INTERVAL} 秒")
    print(f"  按 Ctrl+C 停止")
    print("=" * 60)

    processed = load_processed()
    print(f"  已处理文件: {len(processed)} 个")

    # 首次扫描
    processed = scan_once(processed)

    try:
        while True:
            time.sleep(POLL_INTERVAL)
            processed = scan_once(processed)
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
