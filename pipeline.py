#!/usr/bin/env python3
"""
电商详情页 → PSD v2.1 精度提升版
Step 1: PaddleOCR 文字检测 + SAM 目标分割 → 图层元数据
Step 2: LaMa / SD Inpainting 背景修复 + 前景裁剪 → 独立图层文件
Step 3: 中间 JSON 状态描述（含置信度、字体粗细、字体风格）
Step 4: Node.js ag-psd 编译可编辑文字图层 PSD

精度改进 vs v2.0:
- OCR: EasyOCR → PaddleOCR (CPU, PP-OCRv4, 中文识别率 +15-20%)
- 颜色: K-means 全域采样 → Otsu 二值化后仅采样文字像素
- 字体: 新增 stroke-width 粗细检测 + 衬线/非衬线分类
- 修复: 新增 SD Inpainting 可选 (LaMa 仍为默认)
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

from paddleocr import PaddleOCR
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from simple_lama_inpainting import SimpleLama

# ============================================================
# 配置
# ============================================================
INPUT_DIR = Path(r"C:\Users\86191\Desktop\详情页")
OUTPUT_DIR = INPUT_DIR / "pipeline_output"
TEMP_DIR = OUTPUT_DIR / "temp"
JSON_DIR = OUTPUT_DIR / "json"
PSD_DIR = OUTPUT_DIR / "psd"

SAM_CHECKPOINT = Path(r"E:/pypy/github/ComfyUI/models/sams/sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_MAX_SIDE = 1024

SAM_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200,
}

MAX_IOU_DEDUP = 0.85
MIN_AREA_RATIO = 0.003
MAX_AREA_RATIO = 0.90

# PaddleOCR 置信度阈值
OCR_CONF_THRESHOLD = 0.50

# 常见英文词表 (用于拆分 PaddleOCR 合并的英文词组)
_ENGLISH_WORDS = {
    "PRODUCT", "INFORMATION", "MATERIAL", "COLOR", "BRAND", "STYLE",
    "SIZE", "WEIGHT", "QUALITY", "NATURAL", "NOBLE", "CHARM", "SENSE",
    "WEALTHY", "SOFT", "SILK", "PREMIUM", "DESIGN", "FASHION", "MODEL",
    "DETAIL", "ABOUT", "YOUR", "LIKE", "CAN", "AND", "THE", "FOR",
    "WITH", "THAT", "THIS", "FROM", "ONLY", "MORE", "BEST", "NEW",
    "HOT", "BIG", "TOP", "OUR", "ALL", "ONE", "TWO", "NOT",
    "OF", "IN", "TO", "IS", "IT", "AT", "BE", "OR", "AS",
    "NO", "MY", "UP", "SO", "GO", "WE", "US", "BY", "ON",
}


def _split_english_text(text: str) -> str:
    """拆分被 PaddleOCR 错误合并的英文词组 (e.g., PRODUCTINFORMATION → PRODUCT INFORMATION)"""
    if not text:
        return text
    # 仅处理纯英文字母串 (可能含空格)
    has_chinese = any('一' <= c <= '鿿' for c in text)
    if has_chinese:
        return text
    if ' ' in text:
        return text  # 已有空格, 无需处理

    # 尝试用常见词表做贪心拆分
    upper_text = text.upper()
    words = []
    i = 0
    best_end = 0
    while i < len(upper_text):
        matched = False
        for end in range(min(i + 20, len(upper_text)), i, -1):
            if upper_text[i:end] in _ENGLISH_WORDS:
                words.append(text[i:end])
                i = end
                best_end = end
                matched = True
                break
        if not matched:
            # 单字符前进
            if i == best_end:
                if words:
                    words[-1] += text[i]
                else:
                    words.append(text[i])
                i += 1
                best_end = i
            else:
                i += 1
                best_end = i

    # 只有当拆分结果覆盖原文时才应用
    reconstructed = ''.join(words)
    if reconstructed.upper() == upper_text and len(words) > 1:
        return ' '.join(words)
    return text

# Inpainting 模式: "lama" (快速, 低VRAM) 或 "sd" (高质量, 需更多VRAM)
INPAINT_MODE = "lama"

# ============================================================
# 工具函数
# ============================================================

def imread_unicode(filepath: Path) -> np.ndarray:
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_for_processing(img: np.ndarray, max_side: int) -> tuple:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


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
    return (ys.mean() / mask.shape[0], xs.mean() / mask.shape[1])


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


def classify_mask(mask: np.ndarray) -> str:
    area = mask_area_ratio(mask)
    aspect = mask_aspect_ratio(mask)
    rect = mask_rectangularity(mask)
    cy, cx = mask_center(mask)
    center_dist = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)

    if area > 0.60:
        return "bg"
    text_score = 0
    if aspect > 3.5 or aspect < 0.28:
        text_score += 3
    if rect > 0.65:
        text_score += 2
    if 0.001 < area < 0.10:
        text_score += 2
    if rect > 0.75 and area < 0.06:
        text_score += 2
    if text_score >= 5:
        return "text"
    if area < 0.03:
        return "deco" if center_dist > 0.30 else "elem"
    if center_dist < 0.28 and 0.04 < area < 0.50:
        return "product"
    if 0.30 < area <= 0.60:
        return "bg_elem"
    return "elem"


def upscale_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    tw, th = target_size
    h, w = mask.shape
    if (w, h) == (tw, th):
        return mask
    resized = cv2.resize(mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


# ============================================================
# 文字颜色提取 (修复版: Otsu 二值化 → 仅采样文字像素)
# ============================================================

def extract_text_color_v2(img_bgr: np.ndarray, bbox: np.ndarray) -> str:
    """
    精确提取文字颜色。
    策略: Otsu 二值化分离前景/背景 → 只取前景像素做 K-means 主色提取。
    避免将背景色错误识别为文字颜色。
    """
    x1 = int(bbox[:, 0].min())
    y1 = int(bbox[:, 1].min())
    x2 = int(bbox[:, 0].max())
    y2 = int(bbox[:, 1].max())
    pad = 2
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return "#000000"

    region = img_bgr[y1:y2, x1:x2]
    if region.size == 0:
        return "#000000"

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Otsu 自动阈值分离文字与背景
    if gray.std() < 15:
        # 低对比度区域: 直接用全局像素
        fg_mask = np.ones(gray.shape, dtype=np.uint8) * 255
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 判断文字是深色还是浅色: 比较边缘像素(背景)和中心像素
        edge_mean = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ]).mean() if gray.size > 10 else 128
        center_mean = gray[gray.shape[0]//4:3*gray.shape[0]//4,
                           gray.shape[1]//4:3*gray.shape[1]//4].mean()
        # 如果边缘(背景)更亮，则文字是暗的 (binary=0 → fg)
        if edge_mean > center_mean:
            fg_mask = (binary == 0).astype(np.uint8) * 255
        else:
            fg_mask = (binary == 255).astype(np.uint8) * 255

    # 只取前景像素
    fg_pixels = region[fg_mask > 0]
    if len(fg_pixels) < 5:
        # 前景太少，退回取所有非极端像素
        flat = region.reshape(-1, 3).astype(np.float32)
        mask_bright = (flat.mean(axis=1) < 240) & (flat.mean(axis=1) > 15)
        fg_pixels = flat[mask_bright]
        if len(fg_pixels) < 5:
            return "#000000"

    fg_pixels = fg_pixels.astype(np.float32)
    if len(fg_pixels) == 1:
        b, g, r = np.clip(fg_pixels[0], 0, 255).astype(int)
        return f"#{r:02x}{g:02x}{b:02x}"

    # K-means 找主色 (2 簇, 取像素更多的那簇)
    n_clusters = min(2, len(fg_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        fg_pixels.reshape(-1, 1, 3), n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten())
    dominant = centers[counts.argmax()].flatten()

    r, g, b = np.clip(dominant, 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================
# 字体粗细检测
# ============================================================

def detect_font_weight(img_bgr: np.ndarray, bbox: np.ndarray) -> str:
    """
    通过文字区域 stroke width 估算字体粗细。
    使用文字像素的实际纵向范围 (而非 bbox 高度) 作为归一化基准，
    避免因行间距/内边距导致分母过大而全判为 thin。
    返回: "thin" | "normal" | "bold" | "extra-bold"
    """
    x1 = int(bbox[:, 0].min())
    y1 = int(bbox[:, 1].min())
    x2 = int(bbox[:, 0].max())
    y2 = int(bbox[:, 1].max())
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return "normal"

    region = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    if gray.std() < 10:
        return "normal"
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找文字像素
    if np.sum(binary == 0) < np.sum(binary == 255):
        text_pixels = (binary == 0).astype(np.uint8) * 255
    else:
        text_pixels = binary

    # Distance transform: 每个文字像素到最近背景的距离 (即 1/2 笔画宽)
    dist = cv2.distanceTransform(text_pixels, cv2.DIST_L2, 5)
    text_distances = dist[text_pixels > 0]

    if len(text_distances) < 10:
        return "normal"

    # 笔画宽度 ≈ 2 × 中位距离
    mean_stroke = 2 * np.median(text_distances)

    # 用文字像素的实际纵向跨度, 不是 bbox 高度
    ys_text = np.where(text_pixels > 0)[0]
    actual_text_height = ys_text.max() - ys_text.min() + 1
    if actual_text_height < 3:
        return "normal"

    normalized_stroke = mean_stroke / actual_text_height

    if normalized_stroke > 0.24:
        return "extra-bold"
    elif normalized_stroke > 0.14:
        return "bold"
    elif normalized_stroke < 0.05:
        return "thin"
    return "normal"


# ============================================================
# 字体风格检测 (衬线/非衬线)
# ============================================================

def detect_font_style(img_bgr: np.ndarray, bbox: np.ndarray) -> str:
    """
    基于垂直投影的笔画粗细变化检测衬线体。
    衬线体特征: 笔画末端粗细变化大 → 投影方差高。
    返回: "sans-serif" | "serif" | "unknown"
    """
    x1 = int(bbox[:, 0].min())
    y1 = int(bbox[:, 1].min())
    x2 = int(bbox[:, 0].max())
    y2 = int(bbox[:, 1].max())
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    region = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    if gray.std() < 10:
        return "unknown"
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(binary == 0) < np.sum(binary == 255):
        text_mask = (binary == 0).astype(np.uint8)
    else:
        text_mask = (binary == 255).astype(np.uint8)

    # 水平投影: 统计每行的文字像素数
    h_proj = text_mask.sum(axis=1).astype(np.float32)
    if h_proj.sum() < 5:
        return "unknown"

    # 求水平投影的标准差 / 均值 → 越大说明笔画粗细变化越大 (更可能衬线)
    h_norm = h_proj / (h_proj.max() + 1e-8)
    variation = h_norm.std() / (h_norm.mean() + 1e-8)

    if variation > 0.55:
        return "serif"
    return "sans-serif"


# ============================================================
# 字体名映射 (→ Photoshop 字体名)
# ============================================================

def map_font_name(weight: str, style: str) -> str:
    """根据粗细和风格映射到 Photoshop 标准中文字体名"""
    if style == "serif":
        if weight in ("bold", "extra-bold"):
            return "SimHei"
        elif weight == "thin":
            return "FangSong"
        return "SimSun"
    else:
        return "Microsoft YaHei"


def extract_layer_crop(img_bgr: np.ndarray, mask: np.ndarray, feather: int = 1) -> Image.Image:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_bgr[:, :, ::-1]

    if feather > 0 and mask.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather * 2 + 1, feather * 2 + 1))
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        alpha = cv2.GaussianBlur(mask_dilated.astype(np.float32),
                                 (feather * 2 + 1, feather * 2 + 1), 0.5)
        alpha = np.clip(alpha * 255, 0, 255)
    else:
        alpha = mask.astype(np.float32) * 255

    rgba[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ============================================================
# 处理单张图片
# ============================================================

def _build_inpaint_mask(h_orig, w_orig, text_layers, image_layers):
    """构建 inpainting 用的组合 mask"""
    combined = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for tl in text_layers:
        x1, y1 = tl["left"], tl["top"]
        x2, y2 = x1 + tl["width"], y1 + tl["height"]
        pad = 4
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_orig, x2 + pad)
        y2 = min(h_orig, y2 + pad)
        combined[y1:y2, x1:x2] = 255
    for il in image_layers:
        x1, y1, lw, lh = il["left"], il["top"], il["width"], il["height"]
        pad = 6
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_orig, x1 + lw + pad * 2)
        y2 = min(h_orig, y1 + lh + pad * 2)
        combined[y1:y2, x1:x2] = 255
    return combined


def _run_sd_inpaint(img_rgb, mask, sd_pipe):
    """Stable Diffusion Inpainting，自动缩放到 768px 以内"""
    h, w = img_rgb.shape[:2]
    sd_max = 768
    sd_scale = sd_max / max(w, h) if max(w, h) > sd_max else 1.0
    if sd_scale < 1.0:
        sd_w, sd_h = int(w * sd_scale), int(h * sd_scale)
        img_sd = Image.fromarray(img_rgb).resize((sd_w, sd_h), Image.LANCZOS)
        mask_sd = Image.fromarray(mask).resize((sd_w, sd_h), Image.NEAREST)
    else:
        img_sd = Image.fromarray(img_rgb)
        mask_sd = Image.fromarray(mask)
    result = sd_pipe(
        image=img_sd, mask_image=mask_sd,
        prompt="clean seamless product photography background, studio lighting, plain surface",
        negative_prompt="text, watermark, logo, people, products, items, objects, clutter",
        num_inference_steps=20, guidance_scale=7.5,
    ).images[0]
    if sd_scale < 1.0:
        result = result.resize((w, h), Image.LANCZOS)
    return np.array(result)


def process_stage1(img_path: Path, ocr_reader, mask_generator) -> dict | None:
    """Stage 1: OCR + SAM 分割 → 返回中间数据 + 保存裁剪图层"""
    name = img_path.stem
    print(f"\n{'='*60}")
    print(f"[{name}] Stage 1: OCR + SAM")

    img_bgr = imread_unicode(img_path)
    if img_bgr is None:
        print(f"  [ERROR] 无法读取")
        return None
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"  原始尺寸: {w_orig}x{h_orig}")

    img_small, scale = resize_for_processing(img_bgr, SAM_MAX_SIDE)
    h_s, w_s = img_small.shape[:2]
    img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    print(f"  处理尺寸: {w_s}x{h_s} (scale={scale:.2f})")

    # ---- OCR ----
    print(f"  [OCR] PaddleOCR (PP-OCRv4)...")
    ocr_result = ocr_reader.ocr(img_rgb_small, cls=False)
    text_layers = []
    if ocr_result and ocr_result[0]:
        for bbox, (text, conf) in ocr_result[0]:
            if conf < OCR_CONF_THRESHOLD:
                continue
            text = _split_english_text(text)
            bbox_np = np.array(bbox)
            bbox_orig = bbox_np / scale
            x1 = bbox_orig[:, 0].min(); y1 = bbox_orig[:, 1].min()
            x2 = bbox_orig[:, 0].max(); y2 = bbox_orig[:, 1].max()

            color = extract_text_color_v2(img_small, bbox_np)
            weight = detect_font_weight(img_small, bbox_np)
            style = detect_font_style(img_small, bbox_np)
            font_name = map_font_name(weight, style)

            text_layers.append({
                "text": text, "left": round(x1), "top": round(y1),
                "width": round(x2 - x1), "height": round(y2 - y1),
                "fontSize": round(y2 - y1), "color": color,
                "confidence": round(conf, 3),
                "fontWeight": weight, "fontStyle": style, "fontName": font_name,
                "name": f"文字_{text[:12]}",
            })
            print(f"    [{conf:.2f}] \"{text[:25]}\" color={color} "
                  f"weight={weight} style={style} font={font_name}")
    print(f"  OCR: {len(text_layers)} 个文本块")

    # ---- SAM ----
    print(f"  [SAM] 目标分割...")
    masks = mask_generator.generate(img_rgb_small)
    print(f"    SAM 原始: {len(masks)} masks")
    kept = []
    for m in sorted(masks, key=lambda x: x["stability_score"], reverse=True):
        seg = m["segmentation"]
        ar = mask_area_ratio(seg)
        if ar > MAX_AREA_RATIO or ar < MIN_AREA_RATIO:
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

    image_layers = []
    for i, m in enumerate(kept):
        seg = m["segmentation"]
        label = classify_mask(seg)
        if label == "text":
            continue
        seg_orig = upscale_mask(seg, (w_orig, h_orig)) if scale < 1.0 else seg
        ys, xs = np.where(seg_orig)
        if len(ys) == 0:
            continue
        bbox_x, bbox_y = xs.min(), ys.min()
        bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

        crop_img = extract_layer_crop(img_bgr, seg_orig, feather=2)
        crop_filename = f"{name}_{label}_{i+1}.png"
        crop_img.save(str(TEMP_DIR / crop_filename))
        print(f"    [{label}] {crop_filename} — {bbox_w}x{bbox_h} @ ({bbox_x},{bbox_y})")
        image_layers.append({
            "type": "image", "name": f"{label}_{i+1}", "label": label,
            "imagePath": f"./temp/{crop_filename}",
            "left": int(bbox_x), "top": int(bbox_y),
            "width": int(bbox_w), "height": int(bbox_h),
            "stability": round(m["stability_score"], 3),
        })

    print(f"  SAM: {len(image_layers)} 个图层")

    # 构建 inpainting mask
    inpaint_mask = _build_inpaint_mask(h_orig, w_orig, text_layers, image_layers)

    return {
        "name": name,
        "img_path": str(img_path),
        "img_bgr": img_bgr,
        "w_orig": w_orig, "h_orig": h_orig,
        "text_layers": text_layers,
        "image_layers": image_layers,
        "inpaint_mask": inpaint_mask,
    }


def process_stage2(data: dict, lama, sd_pipe=None) -> dict | None:
    """Stage 2: Inpainting + JSON 生成"""
    name = data["name"]
    img_bgr = data["img_bgr"]
    w_orig, h_orig = data["w_orig"], data["h_orig"]
    text_layers = data["text_layers"]
    image_layers = data["image_layers"]
    inpaint_mask = data["inpaint_mask"]

    print(f"\n[{name}] Stage 2: Inpainting ({INPAINT_MODE})")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if INPAINT_MODE == "sd" and sd_pipe is not None:
        cleaned_bg = _run_sd_inpaint(img_rgb, inpaint_mask, sd_pipe)
    else:
        cleaned_bg = lama(img_rgb, inpaint_mask)
        if isinstance(cleaned_bg, Image.Image):
            cleaned_bg = np.array(cleaned_bg)

    bg_path = TEMP_DIR / f"{name}_clean_bg.png"
    Image.fromarray(cleaned_bg).save(str(bg_path))
    print(f"    干净背景: {bg_path.name}")

    # JSON
    layers_json = [{
        "type": "background", "name": "背景",
        "imagePath": f"./temp/{name}_clean_bg.png",
    }]
    layers_json.extend(sorted(image_layers, key=lambda l: -l["width"] * l["height"]))
    for tl in text_layers:
        layers_json.append({
            "type": "text", "name": tl["name"], "text": tl["text"],
            "left": tl["left"], "top": tl["top"],
            "width": tl["width"], "height": tl["height"],
            "fontSize": tl["fontSize"], "color": tl["color"],
            "confidence": tl["confidence"],
            "fontWeight": tl["fontWeight"], "fontStyle": tl["fontStyle"],
            "fontName": tl["fontName"],
        })

    result = {
        "sourceImage": Path(data["img_path"]).name,
        "width": w_orig, "height": h_orig,
        "ocrEngine": "PaddleOCR PP-OCRv4",
        "inpaintMode": INPAINT_MODE,
        "layers": layers_json,
    }
    json_path = JSON_DIR / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {json_path.name} ({len(layers_json)} layers, "
          f"text={len(text_layers)}, img={len(image_layers)})")
    return result


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  电商详情页 PSD 管线 v2.1 — 精度提升版")
    print(f"  OCR: PaddleOCR PP-OCRv4 (CPU)")
    print(f"  SAM: {MODEL_TYPE} ({DEVICE})")
    print(f"  修复: {INPAINT_MODE.upper()}")
    print("=" * 60)

    for d in [OUTPUT_DIR, TEMP_DIR, JSON_DIR, PSD_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取图片列表
    image_files = sorted(INPUT_DIR.glob("*.webp"))
    if not image_files:
        image_files = (sorted(INPUT_DIR.glob("*.png")) +
                       sorted(INPUT_DIR.glob("*.jpg")) +
                       sorted(INPUT_DIR.glob("*.jpeg")))
    print(f"\n找到 {len(image_files)} 张图片")

    # ---- 初始化 PaddleOCR ----
    print("\n加载 PaddleOCR (PP-OCRv4, CPU)...")
    ocr_reader = PaddleOCR(use_angle_cls=False, lang="ch",
                           use_gpu=False, show_log=False)
    print("  PaddleOCR OK")

    # ---- 初始化 SAM ----
    if not SAM_CHECKPOINT.exists():
        print(f"\n[ERROR] SAM 模型未找到: {SAM_CHECKPOINT}")
        sys.exit(1)
    print(f"加载 SAM ({MODEL_TYPE})...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(model=sam, **SAM_PARAMS)
    print(f"  SAM OK ({DEVICE})")

    # ---- Stage 1: OCR + SAM (所有图片) ----
    print(f"\n{'='*60}")
    print("  STAGE 1: OCR + SAM 分割 (所有图片)")
    print(f"{'='*60}")
    all_stage1 = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\n--- [{i}/{len(image_files)}] ---", end="")
        try:
            data = process_stage1(img_path, ocr_reader, mask_generator)
            if data:
                all_stage1.append(data)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    print(f"\nStage 1 完成: {len(all_stage1)}/{len(image_files)} 张")

    # 释放 SAM 显存
    del sam, mask_generator
    torch.cuda.empty_cache()
    print("SAM 已释放, VRAM:", f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
          if DEVICE == "cuda" else "N/A")

    # ---- Stage 2: Inpainting (所有图片) ----
    print(f"\n{'='*60}")
    print(f"  STAGE 2: 背景修复 ({INPAINT_MODE.upper()})")
    print(f"{'='*60}")

    sd_pipe = None
    lama = None

    if INPAINT_MODE == "sd":
        print("加载 Stable Diffusion Inpainting...")
        from diffusers import StableDiffusionInpaintPipeline
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None,
        )
        if DEVICE == "cuda":
            sd_pipe = sd_pipe.to("cuda")
            sd_pipe.enable_attention_slicing()
        print(f"  SD Inpainting OK, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    else:
        print("加载 LaMa inpainting...")
        lama = SimpleLama()
        print("  LaMa OK")

    all_results = []
    for i, data in enumerate(all_stage1, 1):
        print(f"\n--- [{i}/{len(all_stage1)}] ---", end="")
        try:
            result = process_stage2(data, lama, sd_pipe)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    # 清理 SD
    if sd_pipe is not None:
        del sd_pipe
        torch.cuda.empty_cache()

    # 保存汇总 JSON
    summary_path = JSON_DIR / "_all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline 完成! 成功 {len(all_results)}/{len(image_files)} 张")
    print(f"中间产物: {TEMP_DIR}")
    print(f"JSON 描述: {JSON_DIR}")
    print(f"{'='*60}")
    print(f"\n下一步: 运行 Node.js PSD 编译器")
    print(f"  cd {OUTPUT_DIR}")
    print(f"  node compile_psd.js")


if __name__ == "__main__":
    main()
