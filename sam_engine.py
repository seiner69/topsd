"""SAM 目标分割引擎"""
import sys

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from config import (
    SAM_CHECKPOINT, MODEL_TYPE, DEVICE, SAM_PARAMS,
    MAX_IOU_DEDUP, MIN_AREA_RATIO, MAX_AREA_RATIO, SAM_MAX_SIDE,
    TEMP_DIR,
)
from image_utils import (
    compute_iou, mask_area_ratio, mask_center,
    mask_aspect_ratio, mask_rectangularity, upscale_mask,
    extract_layer_crop, resize_for_processing,
)


def classify_mask(mask: np.ndarray) -> str:
    """基于几何特征将 mask 分类为 product / text / deco / elem / bg / bg_elem"""
    area = mask_area_ratio(mask)
    aspect = mask_aspect_ratio(mask)
    rect = mask_rectangularity(mask)
    cy, cx = mask_center(mask)
    center_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

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


class SAMEngine:
    """SAM 分割封装"""

    def __init__(self):
        if not SAM_CHECKPOINT.exists():
            print(f"\n[ERROR] SAM 模型未找到: {SAM_CHECKPOINT}")
            sys.exit(1)

        self._sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
        self._sam.to(device=DEVICE)
        self._sam.eval()
        self._generator = SamAutomaticMaskGenerator(model=self._sam, **SAM_PARAMS)

    def release(self):
        del self._sam, self._generator
        torch.cuda.empty_cache()

    def segment(self, img_bgr: np.ndarray, name: str) -> list[dict]:
        """
        对图片运行 SAM 分割, 过滤/去重/分类 mask, 保存裁剪图层。
        返回 image_layers 列表。
        """
        # 缩放到处理尺寸
        h_orig, w_orig = img_bgr.shape[:2]
        img_small, scale = resize_for_processing(img_bgr, SAM_MAX_SIDE)
        img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # noqa: F821

        masks = self._generator.generate(img_rgb_small)
        kept = self._dedup_masks(masks)

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

            crop = extract_layer_crop(img_bgr, seg_orig, feather=2)
            crop_filename = f"{name}_{label}_{i + 1}.png"
            crop.save(str(TEMP_DIR / crop_filename))

            image_layers.append({
                "type": "image", "name": f"{label}_{i + 1}", "label": label,
                "imagePath": f"./output/temp/{crop_filename}",
                "left": int(bbox_x), "top": int(bbox_y),
                "width": int(bbox_w), "height": int(bbox_h),
                "stability": round(m["stability_score"], 3),
            })

        return image_layers

    @staticmethod
    def _dedup_masks(masks: list) -> list:
        """按稳定性排序, 过滤面积异常 + IoU 去重"""
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
        return kept
