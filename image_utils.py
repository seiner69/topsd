"""图像 I/O 与 mask 运算工具"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def imread_unicode(filepath: Path) -> np.ndarray:
    """读取任意编码路径的图片"""
    with open(filepath, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_for_processing(img: np.ndarray, max_side: int) -> tuple:
    """缩放到指定最大边长, 返回 (image, scale)"""
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


def upscale_mask(mask: np.ndarray, target_size: tuple) -> np.ndarray:
    """将 mask 映射回原图尺寸"""
    tw, th = target_size
    h, w = mask.shape
    if (w, h) == (tw, th):
        return mask
    resized = cv2.resize(mask.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def extract_layer_crop(img_bgr: np.ndarray, mask: np.ndarray, feather: int = 1) -> Image.Image:
    """将 mask 区域从原图裁剪为带 alpha 通道的 RGBA 图像"""
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
