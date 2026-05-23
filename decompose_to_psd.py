#!/usr/bin/env python3
"""
电商详情页图片 → PSD 全图层拆解
基于 SAM (Segment Anything Model) 自动分割所有视觉元素，
每个检测到的元素导出为独立的 PSD 图层。

策略: 先缩小图片做 SAM 分割 (省显存), 再映射 mask 回原始分辨率做 PSD。
"""

import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from psd_tools import PSDImage

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================

INPUT_DIR = Path(r"C:\Users\86191\Desktop\详情页")
SAM_CHECKPOINT = Path(r"E:/pypy/github/ComfyUI/models/sams/sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM 处理时的最大边长 (节省显存)
SAM_MAX_SIDE = 1024

# SAM 自动分割参数
SAM_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 0,            # 禁用裁剪层 (大图已缩小)
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200,
}

# mask 过滤参数
MAX_IOU_DEDUP = 0.85
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.92
MAX_LAYERS = 30

# ============================================================
# 工具函数
# ============================================================

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def mask_area_ratio(mask: np.ndarray) -> float:
    return mask.sum() / mask.size


def mask_center(mask: np.ndarray) -> tuple:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.5, 0.5)
    h, w = mask.shape
    return (ys.mean() / h, xs.mean() / w)


def mask_aspect_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 1.0
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return w / h if h > 0 else 1.0


def mask_rectangularity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return mask.sum() / (h * w)


def classify_mask_smart(mask: np.ndarray) -> str:
    """根据形状和位置特征分类 mask。返回中文标签。"""
    area_r = mask_area_ratio(mask)
    aspect = mask_aspect_ratio(mask)
    rect = mask_rectangularity(mask)
    cy, cx = mask_center(mask)
    center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    if area_r > 0.65:
        return "bg"

    # 文字特征
    text_score = 0
    if aspect > 3.5 or aspect < 0.28:
        text_score += 3
    if rect > 0.65:
        text_score += 2
    if 0.001 < area_r < 0.10:
        text_score += 2
    if rect > 0.75 and area_r < 0.06:
        text_score += 2
    if text_score >= 5:
        return "text"

    if area_r < 0.03:
        return "deco" if center_dist > 0.30 else "elem"

    if center_dist < 0.28 and 0.04 < area_r < 0.50:
        return "product"

    if 0.30 < area_r <= 0.65:
        return "bg_elem"

    return "elem"


def imread_unicode(filepath: Path) -> np.ndarray:
    """读取图片 (支持中文路径)"""
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {filepath}")
    return img


def resize_for_sam(img: np.ndarray, max_side: int = 1024) -> tuple:
    """
    将图片等比缩放到适合 SAM 处理的尺寸。
    返回: (缩放后的RGB图片, 缩放因子, 原始尺寸)
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0, (w, h)

    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale, (w, h)


def upscale_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """将 mask 缩放到目标尺寸 (最近邻插值保持二值)"""
    tw, th = target_size
    h, w = mask.shape
    if (w, h) == (tw, th):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def upscale_mask_smooth(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """将 mask 缩放到目标尺寸 (双线性插值 + 阈值, 边缘更平滑)"""
    tw, th = target_size
    h, w = mask.shape
    if (w, h) == (tw, th):
        return mask
    resized = cv2.resize(mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def extract_layer(image_bgr: np.ndarray, mask: np.ndarray, feather: int = 2) -> Image.Image:
    """从原图中提取 mask 区域像素，生成 RGBA 图层。feather: 羽化半径"""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if feather > 0 and mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather * 2 + 1, feather * 2 + 1))
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        alpha = mask_dilated.astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), feather / 2)
        alpha = np.clip(alpha, 0, 1)
    else:
        alpha = mask.astype(np.float32)

    rgba[:, :, :3] = image_bgr[:, :, ::-1]  # BGR → RGB
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ============================================================
# 核心处理
# ============================================================

def process_image(image_path: Path, mask_generator: SamAutomaticMaskGenerator, output_dir: Path):
    """处理单张图片，生成 PSD"""
    name = image_path.stem
    print(f"\n{'='*60}")
    print(f"处理: {image_path.name}")

    # 加载原始图片
    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        print(f"  [ERROR] 无法读取: {image_path}")
        return
    h_orig, w_orig = image_bgr.shape[:2]
    print(f"  原始尺寸: {w_orig}x{h_orig}, {w_orig*h_orig/1e6:.1f}MP")

    # 缩放用于 SAM
    image_small, scale, _ = resize_for_sam(image_bgr, SAM_MAX_SIDE)
    h_small, w_small = image_small.shape[:2]
    image_rgb_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    print(f"  SAM 处理尺寸: {w_small}x{h_small} (缩放 {scale:.2f}x)")

    # ---- SAM 自动分割 ----
    print(f"  运行 SAM 自动分割...")
    masks = mask_generator.generate(image_rgb_small)
    print(f"  SAM 生成了 {len(masks)} 个原始 mask")

    if len(masks) == 0:
        print(f"  [WARN] 未检测到任何 mask，跳过")
        return

    # ---- 过滤 mask ----
    masks_sorted = sorted(masks, key=lambda m: m["stability_score"], reverse=True)
    kept = []
    for m in masks_sorted:
        seg = m["segmentation"]
        area_r = mask_area_ratio(seg)
        if area_r > MAX_AREA_RATIO or area_r < MIN_AREA_RATIO:
            continue
        is_dup = False
        for existing in kept:
            if compute_iou(seg, existing["segmentation"]) > MAX_IOU_DEDUP:
                if m["stability_score"] > existing["stability_score"]:
                    existing.update(m)
                is_dup = True
                break
        if not is_dup:
            kept.append(m)

    print(f"  去重后保留 {len(kept)} 个 mask")

    # ---- 分类并生成原始分辨率图层 ----
    layers_info = []
    target_size = (w_orig, h_orig)

    for i, m in enumerate(kept):
        seg_small = m["segmentation"]
        label = classify_mask_smart(seg_small)
        area_r = mask_area_ratio(seg_small)

        # 映射 mask 回原始分辨率
        if scale < 1.0:
            seg_orig = upscale_mask_smooth(seg_small, target_size)
        else:
            seg_orig = seg_small

        # 从原图提取图层
        layer_img = extract_layer(image_bgr, seg_orig, feather=2)
        ys, xs = np.where(seg_orig)
        bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        area_r_orig = mask_area_ratio(seg_orig)

        layers_info.append({
            "index": i,
            "name": f"{label}_{i+1}",
            "label": label,
            "image": layer_img,
            "bbox": bbox,
            "area_ratio": area_r_orig,
            "stability": m["stability_score"],
        })

    # ---- 排序 ----
    label_order = {"bg": 0, "bg_elem": 1, "product": 2, "elem": 3, "text": 4, "deco": 5}
    layers_info.sort(key=lambda l: (label_order.get(l["label"], 3), -l["area_ratio"]))

    # ---- 限制图层数 ----
    if len(layers_info) > MAX_LAYERS:
        by_label = defaultdict(list)
        for l in layers_info:
            by_label[l["label"]].append(l)
        layers_info = []
        for label in ["bg", "bg_elem", "product", "text", "deco", "elem"]:
            items = sorted(by_label[label], key=lambda l: -l["stability"])[:6]
            layers_info.extend(items)
        layers_info.sort(key=lambda l: -l["area_ratio"])
        layers_info = layers_info[:MAX_LAYERS]
        layers_info.sort(key=lambda l: (label_order.get(l["label"], 3), -l["area_ratio"]))

    print(f"  最终图层数: {len(layers_info)}")
    for l in layers_info:
        print(f"    [{l['label']}] {l['name']} — 面积 {l['area_ratio']*100:.1f}%, 稳定性 {l['stability']:.3f}")

    # ---- 创建 PSD ----
    psd = PSDImage.new(mode="RGBA", size=(w_orig, h_orig))

    for layer_info in layers_info:
        x, y, lw, lh = layer_info["bbox"]
        full_canvas = Image.new("RGBA", (w_orig, h_orig), (0, 0, 0, 0))
        full_canvas.paste(layer_info["image"], (x, y))
        psd.create_pixel_layer(full_canvas, name=layer_info["name"])

    output_path = output_dir / f"{name}.psd"
    psd.save(str(output_path))
    print(f"  ✓ 已保存: {output_path.name} ({output_path.stat().st_size/1024:.0f} KB)")
    return output_path


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  电商图片 → PSD 全图层拆解工具")
    print("  基于 Meta SAM (Segment Anything Model)")
    print(f"  SAM 处理分辨率: {SAM_MAX_SIDE}px, 设备: {DEVICE}")
    print("=" * 60)

    if not SAM_CHECKPOINT.exists():
        print(f"\n[ERROR] SAM 模型未找到: {SAM_CHECKPOINT}")
        print(f"请下载: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"或放置到: {SAM_CHECKPOINT.parent}")
        sys.exit(1)

    # 加载 SAM
    print(f"\n加载 SAM 模型 ({MODEL_TYPE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(model=sam, **SAM_PARAMS)
    print(f"  设备: {DEVICE}")

    output_dir = INPUT_DIR / "PSD输出"
    output_dir.mkdir(exist_ok=True)

    # 查找图片
    image_files = sorted(INPUT_DIR.glob("*.webp"))
    if not image_files:
        image_files = (sorted(INPUT_DIR.glob("*.png")) +
                       sorted(INPUT_DIR.glob("*.jpg")) +
                       sorted(INPUT_DIR.glob("*.jpeg")))
    print(f"\n找到 {len(image_files)} 张图片")
    for f in image_files:
        print(f"  - {f.name}")

    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        try:
            result = process_image(img_path, mask_generator, output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"完成! 成功处理 {len(results)}/{len(image_files)} 张图片")
    print(f"PSD 文件保存在: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
