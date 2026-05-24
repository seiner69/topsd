"""SAM 目标分割引擎 - 高精剪影与层级过滤版"""
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

    def segment(self, img_bgr: np.ndarray, name: str) -> tuple[list[dict], np.ndarray]:
        """
        对图片运行 SAM 分割, 过滤/去重/分类 mask, 保存裁剪图层。
        新版特性: 
        1. 过滤冗余子图层 (Nested Mask Suppression)
        2. 返回 image_layers 以及用于背景修复的【高精度联合精确剪影 Mask】 (precise_image_mask)
        """
        h_orig, w_orig = img_bgr.shape[:2]
        img_small, scale = resize_for_processing(img_bgr, SAM_MAX_SIDE)
        img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        masks = self._generator.generate(img_rgb_small)
        kept = self._dedup_masks(masks)

        image_layers = []
        # 初始化一个精确的高分辨率联合 mask，用于 Stage 2 背景缝补
        precise_image_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        for i, m in enumerate(kept):
            seg = m["segmentation"]
            label = classify_mask(seg)
            if label == "text" or label == "bg":
                continue

            seg_orig = upscale_mask(seg, (w_orig, h_orig)) if scale < 1.0 else seg
            
            # 将当前前景物体的精细边缘加入到联合修复 mask 中
            precise_image_mask = np.logical_or(precise_image_mask, seg_orig)

            ys, xs = np.where(seg_orig)
            if len(ys) == 0:
                continue
            bbox_x, bbox_y = xs.min(), ys.min()
            bbox_w, bbox_h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

            # 裁剪带透明通道的 RGBA 图层
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

        return image_layers, (precise_image_mask.astype(np.uint8) * 255)

    @staticmethod
    def _dedup_masks(masks: list) -> list:
        """按稳定性排序, 过滤面积异常 + IoU 去重 + 嵌套包含度过滤 (防止子图层过多)"""
        valid_masks = []
        for m in masks:
            seg = m["segmentation"]
            ar = mask_area_ratio(seg)
            if MIN_AREA_RATIO <= ar <= MAX_AREA_RATIO:
                valid_masks.append(m)

        # 优先选择高稳定性且面积大的主要元素
        valid_masks = sorted(valid_masks, key=lambda x: (x["stability_score"], x["area"]), reverse=True)

        kept = []
        for m in valid_masks:
            seg = m["segmentation"]
            is_dup = False
            for existing in kept:
                existing_seg = existing["segmentation"]
                
                # 1. 传统的 IoU 相似度去重
                if compute_iou(seg, existing_seg) > MAX_IOU_DEDUP:
                    is_dup = True
                    break
                
                # 2. 嵌套包含度抑制 (Nested Suppression)
                # 计算当前 mask 与已有大图层的交集面积占自身的比例
                inter = np.logical_and(seg, existing_seg).sum()
                area_self = seg.sum()
                if area_self > 0:
                    containment = inter / area_self
                    # 如果当前图层有 80% 以上面积被另一个大图层包含，说明它是子部件，直接舍弃
                    if containment > 0.80:
                        is_dup = True
                        break

            if not is_dup:
                kept.append(m)
        return kept
