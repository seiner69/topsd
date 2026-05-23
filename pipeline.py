#!/usr/bin/env python3
"""
JPG→PSD v2.2 — GPU 加速版
Stage 1: PaddleOCR + SAM → 图层元数据 + 裁剪 PNG
Stage 2: Inpainting 背景修复 + JSON 编译
Stage 3: Node.js ag-psd → 可编辑文字图层 PSD
"""
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR, JSON_DIR, PSD_DIR,
    DEVICE, OCR_DEVICE, INPAINT_MODE, SAM_MAX_SIDE,
)
from image_utils import imread_unicode, resize_for_processing
from text_engine import TextEngine
from sam_engine import SAMEngine
from inpaint_engine import InpaintEngine, build_inpaint_mask


# ============================================================
# Stage 1: OCR + SAM
# ============================================================

def process_stage1(img_path: Path, text_engine: TextEngine, sam_engine: SAMEngine) -> dict | None:
    name = img_path.stem
    print(f"\n{'=' * 60}")
    print(f"[{name}] Stage 1: OCR + SAM")

    img_bgr = imread_unicode(img_path)
    if img_bgr is None:
        print(f"  [ERROR] 无法读取")
        return None
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"  原始尺寸: {w_orig}x{h_orig}")

    # OCR 在缩略图上
    img_small, scale = resize_for_processing(img_bgr, SAM_MAX_SIDE)
    h_s, w_s = img_small.shape[:2]
    img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    print(f"  处理尺寸: {w_s}x{h_s} (scale={scale:.2f})")

    # OCR
    print(f"  [OCR] PaddleOCR (PP-OCRv5, {OCR_DEVICE})...")
    text_layers = text_engine.detect(img_rgb_small, scale)
    for tl in text_layers:
        print(f"    [{tl['confidence']:.2f}] \"{tl['text'][:25]}\" "
              f"color={tl['color']} weight={tl['fontWeight']} style={tl['fontStyle']} "
              f"font={tl['fontName']}")
    print(f"  OCR: {len(text_layers)} 个文本块")

    # SAM
    print(f"  [SAM] 目标分割...")
    image_layers = sam_engine.segment(img_bgr, name)
    print(f"  SAM: {len(image_layers)} 个图层")

    inpaint_mask = build_inpaint_mask(h_orig, w_orig, text_layers, image_layers)

    return {
        "name": name, "img_path": str(img_path),
        "img_bgr": img_bgr, "w_orig": w_orig, "h_orig": h_orig,
        "text_layers": text_layers, "image_layers": image_layers,
        "inpaint_mask": inpaint_mask,
    }


# ============================================================
# Stage 2: Inpainting + JSON
# ============================================================

def process_stage2(data: dict, inpaint_engine: InpaintEngine) -> dict | None:
    name = data["name"]
    w_orig, h_orig = data["w_orig"], data["h_orig"]
    text_layers = data["text_layers"]
    image_layers = data["image_layers"]

    print(f"\n[{name}] Stage 2: Inpainting ({INPAINT_MODE})")
    bg_path = inpaint_engine.save_clean_bg(
        data["img_bgr"], data["inpaint_mask"], name
    )
    print(f"    干净背景: {Path(bg_path).name}")

    layers_json = [
        {"type": "background", "name": "背景", "imagePath": bg_path}
    ]
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
        "ocrEngine": "PaddleOCR 3.4.1 PP-OCRv5 (GPU)",
        "inpaintMode": INPAINT_MODE,
        "layers": layers_json,
    }
    json_path = JSON_DIR / f"{name}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  JSON: {json_path.name} ({len(layers_json)} layers, "
          f"text={len(text_layers)}, img={len(image_layers)})")
    return result


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  电商详情页 PSD 管线 v2.2 — GPU 加速版")
    print(f"  OCR: PaddleOCR PP-OCRv5 ({OCR_DEVICE})")
    print(f"  SAM: vit_b ({DEVICE})")
    print(f"  修复: {INPAINT_MODE.upper()}")
    print("=" * 60)

    for d in [OUTPUT_DIR, TEMP_DIR, JSON_DIR, PSD_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 图片列表
    images = sorted(INPUT_DIR.glob("*.webp"))
    if not images:
        images = (sorted(INPUT_DIR.glob("*.png")) +
                  sorted(INPUT_DIR.glob("*.jpg")) +
                  sorted(INPUT_DIR.glob("*.jpeg")))
    print(f"\n找到 {len(images)} 张图片")

    # 初始化引擎
    print(f"\n加载 TextEngine (PaddleOCR 3.4.1, {OCR_DEVICE})...")
    text_engine = TextEngine()
    print("  TextEngine OK")

    print(f"加载 SAMEngine (vit_b, {DEVICE})...")
    sam_engine = SAMEngine()
    print("  SAMEngine OK")

    # ---- Stage 1: OCR + SAM ----
    print(f"\n{'=' * 60}")
    print("  STAGE 1: OCR + SAM")
    print(f"{'=' * 60}")
    all_stage1 = []
    for i, img_path in enumerate(images, 1):
        print(f"\n--- [{i}/{len(images)}] ---", end="")
        try:
            data = process_stage1(img_path, text_engine, sam_engine)
            if data:
                all_stage1.append(data)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    print(f"\nStage 1 完成: {len(all_stage1)}/{len(images)} 张")

    # 释放 SAM
    sam_engine.release()
    print("SAM 已释放")

    # ---- Stage 2: Inpainting ----
    print(f"\n{'=' * 60}")
    print(f"  STAGE 2: 背景修复 ({INPAINT_MODE.upper()})")
    print(f"{'=' * 60}")

    print(f"加载 InpaintEngine ({INPAINT_MODE})...")
    inpaint_engine = InpaintEngine()
    print("  InpaintEngine OK")

    all_results = []
    for i, data in enumerate(all_stage1, 1):
        print(f"\n--- [{i}/{len(all_stage1)}] ---", end="")
        try:
            result = process_stage2(data, inpaint_engine)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    inpaint_engine.release()

    # 汇总
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = JSON_DIR / "_all_results.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline 完成! 成功 {len(all_results)}/{len(images)} 张")
    print(f"  中间产物: {TEMP_DIR}")
    print(f"  JSON: {JSON_DIR}")
    print(f"{'=' * 60}")
    print(f"\n下一步: node compile_psd.js")


if __name__ == "__main__":
    main()
