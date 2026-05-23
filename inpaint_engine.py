"""背景修复引擎 (LaMa / Stable Diffusion)"""
import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from config import INPAINT_MODE, DEVICE, TEMP_DIR


class InpaintEngine:
    """背景修复封装"""

    def __init__(self):
        self._sd_pipe = None
        self._lama = None

        if INPAINT_MODE == "sd":
            from diffusers import StableDiffusionInpaintPipeline
            self._sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                safety_checker=None,
            )
            if DEVICE == "cuda":
                self._sd_pipe = self._sd_pipe.to("cuda")
                self._sd_pipe.enable_attention_slicing()
        else:
            self._lama = SimpleLama()

    def release(self):
        if self._sd_pipe is not None:
            del self._sd_pipe
        self._sd_pipe = None
        self._lama = None
        torch.cuda.empty_cache()

    def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """对图片的 mask 区域进行背景修复, 返回 RGB 图像"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self._sd_pipe is not None:
            return _run_sd_inpaint(img_rgb, mask, self._sd_pipe)
        else:
            result = self._lama(img_rgb, mask)
            if isinstance(result, Image.Image):
                result = np.array(result)
            return result

    def save_clean_bg(self, img_bgr: np.ndarray, mask: np.ndarray, name: str) -> str:
        """修复并保存干净背景, 返回文件路径"""
        cleaned = self.inpaint(img_bgr, mask)
        bg_path = TEMP_DIR / f"{name}_clean_bg.png"
        Image.fromarray(cleaned).save(str(bg_path))
        return f"./output/temp/{name}_clean_bg.png"


def build_inpaint_mask(h_orig: int, w_orig: int,
                       text_layers: list[dict],
                       image_layers: list[dict]) -> np.ndarray:
    """组合文字区域和图片区域为 inpainting mask"""
    combined = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for tl in text_layers:
        x1, y1 = tl["left"], tl["top"]
        x2, y2 = x1 + tl["width"], y1 + tl["height"]
        pad = 4
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w_orig, x2 + pad), min(h_orig, y2 + pad)
        combined[y1:y2, x1:x2] = 255
    for il in image_layers:
        x1, y1, lw, lh = il["left"], il["top"], il["width"], il["height"]
        pad = 6
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w_orig, x1 + lw + pad * 2), min(h_orig, y1 + lh + pad * 2)
        combined[y1:y2, x1:x2] = 255
    return combined


def _run_sd_inpaint(img_rgb: np.ndarray, mask: np.ndarray, sd_pipe) -> np.ndarray:
    """SD Inpainting, 自动缩放到 768px"""
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
