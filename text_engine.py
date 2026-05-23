"""PaddleOCR 引擎 + 文字分析 (颜色/字体检测)"""
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR

from config import OCR_DEVICE, OCR_CONF_THRESHOLD, _ENGLISH_WORDS


class TextEngine:
    """PaddleOCR 封装 + 文字属性分析"""

    def __init__(self):
        self._ocr = PaddleOCR(
            lang="ch",
            device=OCR_DEVICE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def detect(self, img_rgb: np.ndarray, scale: float) -> list[dict]:
        """
        对图像执行 OCR 并分析每个文本块的属性。
        返回: [{"text", "left", "top", "width", "height", "fontSize",
                 "color", "confidence", "fontWeight", "fontStyle", "fontName", "name"}, ...]
        """
        results = self._ocr.predict(img_rgb)
        text_layers = []
        if not results or len(results) == 0:
            return text_layers

        result = results[0]
        rec_texts = result.get("rec_texts", []) or []
        rec_scores = result.get("rec_scores", []) or []
        dt_polys = result.get("dt_polys", []) or []

        for text, conf, poly in zip(rec_texts, rec_scores, dt_polys):
            if conf < OCR_CONF_THRESHOLD:
                continue
            text = _split_english_text(text)
            poly = np.array(poly)
            bbox_orig = poly / scale
            x1 = bbox_orig[:, 0].min()
            y1 = bbox_orig[:, 1].min()
            x2 = bbox_orig[:, 0].max()
            y2 = bbox_orig[:, 1].max()

            color = extract_text_color(img_rgb, poly)
            weight = detect_font_weight(img_rgb, poly)
            style = detect_font_style(img_rgb, poly)
            font_name = map_font_name(weight, style)

            text_layers.append({
                "text": text,
                "left": round(x1), "top": round(y1),
                "width": round(x2 - x1), "height": round(y2 - y1),
                "fontSize": round(y2 - y1),
                "color": color,
                "confidence": round(conf, 3),
                "fontWeight": weight, "fontStyle": style, "fontName": font_name,
                "name": f"文字_{text[:12]}",
            })
        return text_layers


# ---- 英文词拆分 ----

def _split_english_text(text: str) -> str:
    """拆分被 OCR 错误合并的英文词组 (e.g., PRODUCTINFORMATION → PRODUCT INFORMATION)"""
    if not text:
        return text
    has_chinese = any('一' <= c <= '鿿' for c in text)
    if has_chinese or ' ' in text:
        return text

    upper_text = text.upper()
    words = []
    i = best_end = 0
    while i < len(upper_text):
        matched = False
        for end in range(min(i + 20, len(upper_text)), i, -1):
            if upper_text[i:end] in _ENGLISH_WORDS:
                words.append(text[i:end])
                i = best_end = end
                matched = True
                break
        if not matched:
            if i == best_end:
                if words:
                    words[-1] += text[i]
                else:
                    words.append(text[i])
                i = best_end = i + 1
            else:
                i += 1
                best_end = i

    reconstructed = ''.join(words)
    if reconstructed.upper() == upper_text and len(words) > 1:
        return ' '.join(words)
    return text


# ---- 文字颜色提取 ----

def _otsu_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化 + 前景方向判断, 返回前景 mask"""
    if gray.std() < 15:
        return np.ones(gray.shape, dtype=np.uint8) * 255

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_pixels = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    edge_mean = edge_pixels.mean() if len(edge_pixels) > 10 else 128
    cy, ch = gray.shape[0] // 4, gray.shape[0] // 2
    cx, cw = gray.shape[1] // 4, gray.shape[1] // 2
    center_mean = gray[cy:cy + ch, cx:cx + cw].mean()

    if edge_mean > center_mean:
        return (binary == 0).astype(np.uint8) * 255  # 深色文字
    else:
        return (binary == 255).astype(np.uint8) * 255  # 浅色文字


def _bbox_region(img: np.ndarray, poly: np.ndarray, pad: int = 2) -> tuple:
    """从 polygon 提取裁剪区域, 返回 (region, x1, y1) 或 None"""
    x1 = int(poly[:, 0].min())
    y1 = int(poly[:, 1].min())
    x2 = int(poly[:, 0].max())
    y2 = int(poly[:, 1].max())
    h_img, w_img = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return region, x1, y1


def extract_text_color(img_bgr: np.ndarray, poly: np.ndarray) -> str:
    """Otsu 二值化分离前景 → K-means 提取文字主色"""
    out = _bbox_region(img_bgr, poly)
    if out is None:
        return "#000000"
    region, _, _ = out
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    fg_mask = _otsu_foreground_mask(gray)

    fg_pixels = region[fg_mask > 0]
    if len(fg_pixels) < 5:
        flat = region.reshape(-1, 3).astype(np.float32)
        mask_bright = (flat.mean(axis=1) < 240) & (flat.mean(axis=1) > 15)
        fg_pixels = flat[mask_bright]
        if len(fg_pixels) < 5:
            return "#000000"

    fg_pixels = fg_pixels.astype(np.float32)
    n_clusters = min(2, len(fg_pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        fg_pixels.reshape(-1, 1, 3), n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten())
    dominant = centers[counts.argmax()].flatten()
    r, g, b = np.clip(dominant, 0, 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---- 字体粗细 ----

def detect_font_weight(img_bgr: np.ndarray, poly: np.ndarray) -> str:
    """Distance Transform 笔画宽度 → thin / normal / bold / extra-bold"""
    out = _bbox_region(img_bgr, poly, pad=0)
    if out is None:
        return "normal"
    region, _, _ = out
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if gray.std() < 10:
        return "normal"

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_pixels = ((binary == 0) if (binary == 0).sum() < (binary == 255).sum() else binary).astype(np.uint8) * 255

    dist = cv2.distanceTransform(text_pixels, cv2.DIST_L2, 5)
    text_distances = dist[text_pixels > 0]
    if len(text_distances) < 10:
        return "normal"

    mean_stroke = 2 * np.median(text_distances)
    ys_text = np.where(text_pixels > 0)[0]
    actual_height = ys_text.max() - ys_text.min() + 1
    if actual_height < 3:
        return "normal"

    norm = mean_stroke / actual_height
    if norm > 0.24:
        return "extra-bold"
    if norm > 0.14:
        return "bold"
    if norm < 0.05:
        return "thin"
    return "normal"


# ---- 字体风格 ----

def detect_font_style(img_bgr: np.ndarray, poly: np.ndarray) -> str:
    """水平投影方差 → sans-serif / serif / unknown"""
    out = _bbox_region(img_bgr, poly, pad=0)
    if out is None:
        return "unknown"
    region, _, _ = out
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if gray.std() < 10:
        return "unknown"

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = ((binary == 0) if (binary == 0).sum() < (binary == 255).sum() else binary).astype(np.uint8)

    h_proj = text_mask.sum(axis=1).astype(np.float32)
    if h_proj.sum() < 5:
        return "unknown"

    h_norm = h_proj / (h_proj.max() + 1e-8)
    variation = h_norm.std() / (h_norm.mean() + 1e-8)
    return "serif" if variation > 0.55 else "sans-serif"


# ---- 字体映射 ----

def map_font_name(weight: str, style: str) -> str:
    """粗细 + 风格 → Photoshop 中文字体名"""
    if style == "serif":
        if weight in ("bold", "extra-bold"):
            return "SimHei"
        if weight == "thin":
            return "FangSong"
        return "SimSun"
    return "Microsoft YaHei"
