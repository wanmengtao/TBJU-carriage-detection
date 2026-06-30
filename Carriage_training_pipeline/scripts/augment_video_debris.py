"""
视频异物插入脚本
将随机异物插入视频帧中，模拟"出现 → 随镜头移动 → 消失"的真实效果。

异物生命周期：
  1. 在随机帧生成异物，设定持续帧数（30-100 帧）
  2. 每帧位置按 (vx, vy) 漂移，模拟无人机飞过
  3. 超过持续帧数或漂移出画面 → 移除
  4. 每帧有概率新生成异物，保持画面中异物数量动态平衡

用法:
  python scripts/augment_video_debris.py \
    --input_video "F:/海港拍摄/俯视✅️/DJI_20260418132756_0006_S.MP4" \
    --output_dir "视频扩充数据集/carriage_rim_debris" \
    --debris_dir raw_data/debris_materials \
    --debris_filter rock \
    --output_name DJI_0006_S_aug.mp4

  # 批量模式
  python scripts/augment_video_debris.py \
    --input_dir "F:/海港拍摄/俯视✅️" \
    --output_dir "视频扩充数据集/carriage_rim_debris" \
    --debris_dir raw_data/debris_materials \
    --debris_filter rock
"""

import os
import sys
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 复用 augment_debris.py 的函数
sys.path.insert(0, str(Path(__file__).parent))
from augment_debris import (
    load_debris_images,
    paste_with_alpha,
    CATEGORY_BASE_SIZES,
)


# ──────────────────────────────────────────────
# 异物生命周期管理
# ──────────────────────────────────────────────

class DebrisItem:
    """一个异物实例，跟踪其生命周期和位置。"""

    def __init__(self, debris_img, x, y, vx, vy, lifetime):
        self.img = debris_img          # PIL Image (RGBA, 已变换)
        self.x = float(x)             # 当前 x (像素)
        self.y = float(y)             # 当前 y (像素)
        self.vx = vx                  # x 方向漂移速度 (px/frame)
        self.vy = vy                  # y 方向漂移速度 (px/frame)
        self.lifetime = lifetime      # 总持续帧数
        self.age = 0                  # 已存活帧数
        self.w, self.h = debris_img.size

    def update(self):
        """更新位置和年龄。"""
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    def is_alive(self, frame_w, frame_h):
        """检查是否仍在画面内且未超时。"""
        if self.age >= self.lifetime:
            return False
        if self.x + self.w < 0 or self.x > frame_w:
            return False
        if self.y + self.h < 0 or self.y > frame_h:
            return False
        return True

    def paste(self, frame_pil):
        """将异物贴到帧上。"""
        paste_x = int(self.x)
        paste_y = int(self.y)
        if paste_x < 0 or paste_y < 0:
            return
        paste_with_alpha(frame_pil, self.img, (paste_x, paste_y))


def poisson_sample(lam=2.5, max_val=5):
    """Poisson 采样，截断到 [1, max_val]。"""
    L = pow(2.71828, -lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    n = max(1, min(k - 1, max_val))
    return n


def create_debris_for_frame(debris_list, img_w, img_h, region=None):
    """为当前帧生成一个新的 DebrisItem 实例。"""
    debris_img, debris_name, debris_cat = random.choice(debris_list)

    # 确定放置区域
    if region:
        rx, ry, rw, rh = region
    else:
        rx, ry, rw, rh = 0, 0, img_w, img_h

    # 随机旋转（0-360°）
    angle = random.uniform(0, 360)
    rotated = debris_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # 确保不超过区域大小
    dw, dh = rotated.size
    max_scale = min(rw / max(dw, 1), rh / max(dh, 1)) * 0.6
    if max_scale < 0.1:
        return None
    scale = random.uniform(0.3, min(max_scale, 1.0))
    new_w = max(int(dw * scale), 5)
    new_h = max(int(dh * scale), 5)
    rotated = rotated.resize((new_w, new_h), Image.LANCZOS)

    # 随机起始位置
    cx = random.uniform(rx + new_w / 2, rx + rw - new_w / 2)
    cy = random.uniform(ry + new_h / 2, ry + rh - new_h / 2)

    # 随机漂移速度（模拟无人机移动）
    speed = random.uniform(2, 5)
    angle_rad = random.uniform(0, 2 * 3.14159)
    vx = speed * (random.choice([-1, 1]))
    vy = speed * random.uniform(-0.5, 0.5)

    # 持续帧数
    lifetime = random.randint(30, 100)

    return DebrisItem(
        debris_img=rotated,
        x=cx - new_w / 2,
        y=cy - new_h / 2,
        vx=vx,
        vy=vy,
        lifetime=lifetime,
    )


# ──────────────────────────────────────────────
# 视频处理
# ──────────────────────────────────────────────

def process_video(
    input_path: str,
    output_path: str,
    debris_list: list,
    debris_lambda: float = 2.5,
    max_debris: int = 5,
    region=None,
    seed=None,
):
    """处理单个视频，插入异物并输出。"""
    if seed is not None:
        random.seed(seed)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  视频: {frame_w}x{frame_h}, {fps:.1f}fps, {total_frames} 帧")

    # 选择编码器
    fourcc = None
    for codec in [cv2.VideoWriter_fourcc(*'mp4v'), cv2.VideoWriter_fourcc(*'avc1'), cv2.VideoWriter_fourcc(*'XVID')]:
        test_writer = cv2.VideoWriter(output_path, codec, fps, (frame_w, frame_h))
        if test_writer.isOpened():
            fourcc = codec
            test_writer.release()
            break
        test_writer.release()

    if fourcc is None:
        print("  [ERROR] 无可用编码器")
        cap.release()
        return False

    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    active_debris = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 更新现有异物
        active_debris = [d for d in active_debris if d.is_alive(frame_w, frame_h)]
        for d in active_debris:
            d.update()

        # 新增异物：保持画面中 1-max_debris 个异物
        if len(active_debris) < max_debris and random.random() < 0.3:
            new_debris = create_debris_for_frame(debris_list, frame_w, frame_h, region)
            if new_debris:
                active_debris.append(new_debris)

        # 贴图
        for d in active_debris:
            d.paste(frame_pil)

        # 写入
        frame_out = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        writer.write(frame_out)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    进度: {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)")

    cap.release()
    writer.release()

    print(f"  完成: {output_path}")
    return True


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="视频异物插入脚本")
    parser.add_argument("--input_video", type=str, default=None, help="单个输入视频路径")
    parser.add_argument("--input_dir", type=str, default=None, help="输入视频目录（批量模式）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--debris_dir", type=str, default="raw_data/debris_materials", help="异物素材目录")
    parser.add_argument("--debris_filter", nargs="*", default=None, help="只使用指定素材子目录")
    parser.add_argument("--output_name", type=str, default=None, help="输出视频文件名（单视频模式）")
    parser.add_argument("--debris_lambda", type=float, default=2.5, help="Poisson λ (default: 2.5)")
    parser.add_argument("--max_debris", type=int, default=5, help="画面中最大异物数 (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")

    args = parser.parse_args()

    if not args.input_video and not args.input_dir:
        print("错误: 请指定 --input_video 或 --input_dir")
        parser.print_help()
        return

    print("=" * 60)
    print("视频异物插入脚本")
    print("=" * 60)

    # 加载异物素材
    debris_dir = Path(args.debris_dir)
    if not debris_dir.is_absolute():
        debris_dir = Path(__file__).parent.parent / debris_dir

    debris_list = load_debris_images(str(debris_dir), args.debris_filter)
    if not debris_list:
        print("错误: 没有可用的异物素材")
        return

    # 收集输入视频
    input_videos = []
    if args.input_video:
        input_videos.append(Path(args.input_video))
    else:
        input_dir = Path(args.input_dir)
        for f in sorted(input_dir.iterdir()):
            if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                input_videos.append(f)

    print(f"\n输入视频: {len(input_videos)} 个")
    print(f"异物素材: {len(debris_list)} 个")
    print(f"Poisson λ: {args.debris_lambda}, 最大异物数: {args.max_debris}")

    # 处理每个视频
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in input_videos:
        print(f"\n处理: {video_path.name}")

        if args.output_name and len(input_videos) == 1:
            out_name = args.output_name
        else:
            stem = video_path.stem
            out_name = f"{stem}_aug.mp4"

        out_path = output_dir / out_name

        random.seed(args.seed)
        process_video(
            input_path=str(video_path),
            output_path=str(out_path),
            debris_list=debris_list,
            debris_lambda=args.debris_lambda,
            max_debris=args.max_debris,
            seed=args.seed,
        )

    print(f"\n{'='*60}")
    print(f"全部完成! 输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
