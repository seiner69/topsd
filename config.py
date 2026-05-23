"""全局配置常量"""
import os
from pathlib import Path

import torch

# ---- 路径 ----
PROJECT_DIR = Path(__file__).parent.resolve()
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
JSON_DIR = OUTPUT_DIR / "json"
PSD_DIR = OUTPUT_DIR / "psd"

SAM_CHECKPOINT = Path(r"E:/pypy/github/ComfyUI/models/sams/sam_vit_b_01ec64.pth")

# ---- 设备 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_DEVICE = "gpu:0" if torch.cuda.is_available() else "cpu"

# ---- SAM 参数 ----
MODEL_TYPE = "vit_b"
SAM_MAX_SIDE = 1024
SAM_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 0,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200,
}

# ---- Mask 过滤 ----
MAX_IOU_DEDUP = 0.85
MIN_AREA_RATIO = 0.003
MAX_AREA_RATIO = 0.90

# ---- OCR ----
OCR_CONF_THRESHOLD = 0.50
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---- Inpainting ----
INPAINT_MODE = "lama"  # "lama" | "sd"

# ---- 英文词表 (OCR 合并词拆分) ----
_ENGLISH_WORDS = {
    "PRODUCT", "INFORMATION", "MATERIAL", "COLOR", "BRAND", "STYLE",
    "SIZE", "WEIGHT", "QUALITY", "NATURAL", "NOBLE", "CHARM", "SENSE",
    "WEALTHY", "SOFT", "SILK", "PREMIUM", "DESIGN", "FASHION", "MODEL",
    "DETAIL", "ABOUT", "YOUR", "LIKE", "CAN", "AND", "THE", "FOR",
    "WITH", "THAT", "THIS", "FROM", "ONLY", "MORE", "BEST", "NEW",
    "HOT", "BIG", "TOP", "OUR", "ALL", "ONE", "TWO", "NOT",
    "OF", "IN", "TO", "IS", "IT", "AT", "BE", "OR", "AS",
    "NO", "MY", "UP", "SO", "GO", "WE", "US", "BY", "ON",
    "ATMOSPHERE", "FULL", "QUALITY", "BETTER", "SMOOTH", "HIGH",
    "HAIR", "SIDE", "BUN", "WEAR", "BEFORE", "AFTER", "EASY",
}
