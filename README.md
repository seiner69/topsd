# JPG→PSD 智能分层管线

电商详情页扁平图片 → 可编辑文字图层的分层 PSD。

## 管线概览

```
input/*.webp (原始图片)
    │
    ▼
Stage 1: 版面分析 (OCR + SAM)
    · PaddleOCR PP-OCRv5 → 文本/坐标/颜色/字体属性
    · SAM 目标分割 → 产品图/装饰元素 mask
    产出: 图层元数据 + 裁剪 PNG + Inpaint mask
    │
    ▼
Stage 2: 背景修复 + JSON 编译
    · LaMa / SD Inpainting → 干净背景
    · 中间 JSON (含置信度/字体元数据)
    产出: JSON + 干净背景 PNG
    │
    ▼
compile_psd.js: PSD 二进制编译
    · ag-psd 写入可编辑文字图层
    · 低置信文字自动丢弃
    产出: output/psd/*.psd
```

## 环境

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11 | |
| Node.js | — | 仅 PSD 编译阶段 |
| torch | 2.5.1+cu121 | CUDA 12.1 |
| paddlepaddle-gpu | 3.2.2 | CUDA 11.8 (与 torch CUDA 12.1 共存) |
| paddleocr | 3.4.1 | PP-OCRv5_server |
| segment-anything | — | vit_b |
| simple-lama-inpainting | 0.1.2 | 默认背景修复 |
| diffusers | 0.38+ | SD Inpainting (可选) |
| ag-psd | — | PSD 二进制写入 |

### 模型文件

| 模型 | 大小 | 路径 |
|------|------|------|
| sam_vit_b_01ec64.pth | 358MB | E:/pypy/github/ComfyUI/models/sams/ |
| PP-OCRv5_server_det/rec | ~20MB | ~/.paddlex/official_models/ (自动下载) |
| big-lama.pt | 196MB | ~/.cache/torch/hub/ (自动下载) |
| SD 1.5 Inpainting | ~5GB | ~/.cache/huggingface/ (首次运行自动下载) |

## 快速开始

```bash
# 1. 安装 Python 依赖
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu==3.2.2 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr==3.4.1
pip install segment-anything simple-lama-inpainting opencv-python Pillow numpy diffusers

# 2. 安装 Node 依赖
npm install

# 3. 放入原始图片到 input/ 目录 (*.webp / *.jpg / *.png)

# 4. 运行管线 → 生成 PSD
python pipeline.py && node compile_psd.js
```

## 目录结构

```
topsd/
├── config.py              # 全局配置 (路径/设备/阈值/词表)
├── pipeline.py            # Stage 1-2 编排层
├── text_engine.py         # PaddleOCR 3.4.1 + 颜色/字体检测
├── sam_engine.py          # SAM vit_b 分割 + mask 分类去重
├── inpaint_engine.py      # LaMa / SD 背景修复
├── image_utils.py         # 图像 I/O 与 mask 运算工具
├── compile_psd.js         # PSD 编译器 (Node.js)
├── package.json
├── README.md
├── CHANGELOG.md
├── input/                 # 输入图片 (用户放入)
│   └── *.webp
└── output/                # 生成产物
    ├── temp/              #   中间 PNG (背景 + 图层裁剪)
    ├── json/              #   JSON 图层描述
    └── psd/               #   最终 PSD
```

## 模块说明

| 模块 | 职责 | 行数 |
|------|------|------|
| `config.py` | 路径、设备、SAM 参数、OCR 阈值、英文词表 | ~55 |
| `image_utils.py` | imread_unicode、缩放、IoU、mask 几何特征、图层裁剪 | ~85 |
| `text_engine.py` | PaddleOCR 封装、Otsu 颜色提取、Distance Transform 粗细检测、字体映射 | ~245 |
| `sam_engine.py` | SAM 分割封装、mask 分类去重、图层导出 | ~125 |
| `inpaint_engine.py` | LaMa/SD 背景修复、inpaint mask 构建 | ~100 |
| `pipeline.py` | 纯编排：两阶段流水线 + 引擎生命周期管理 | ~215 |

## 可调参数

所有参数集中在 `config.py` 顶部：

```python
# 设备
DEVICE = "cuda"           # SAM 设备
OCR_DEVICE = "gpu:0"      # PaddleOCR 设备

# 修复模式
INPAINT_MODE = "lama"     # "lama" | "sd"

# SAM
SAM_MAX_SIDE = 1024
SAM_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 0,
    "min_mask_region_area": 200,
}

# Mask 过滤
MAX_IOU_DEDUP = 0.85
MIN_AREA_RATIO = 0.003
MAX_AREA_RATIO = 0.90

# OCR
OCR_CONF_THRESHOLD = 0.50
```

## JSON 图层描述格式

```json
{
  "sourceImage": "detail.webp",
  "width": 2480, "height": 2132,
  "ocrEngine": "PaddleOCR 3.4.1 PP-OCRv5 (GPU)",
  "inpaintMode": "lama",
  "layers": [
    { "type": "background", "name": "背景", "imagePath": "./output/temp/..." },
    { "type": "image",     "name": "product_2", "label": "product", ... },
    {
      "type": "text", "text": "产品信息",
      "left": 913, "top": 632, "width": 666, "height": 155,
      "fontSize": 155, "color": "#23394e", "confidence": 0.997,
      "fontWeight": "thin", "fontStyle": "sans-serif", "fontName": "Microsoft YaHei"
    }
  ]
}
```

## 精度特性

| 特性 | 技术 | 说明 |
|------|------|------|
| 中文 OCR | PP-OCRv5_server (GPU) | 平均置信度 0.958 |
| 英文 OCR | PP-OCRv5 + 词表拆分 | 自动修复合并词 |
| 文字颜色 | Otsu 二值化 + 前景 K-means | 精确提取，不受背景干扰 |
| 字体粗细 | Distance Transform | thin / normal / bold / extra-bold |
| 字体风格 | 水平投影方差 | sans-serif / serif |
| 字体映射 | 粗细+风格→PS 字体 | SimHei / Microsoft YaHei / FangSong / SimSun |
| 默认修复 | LaMa Inpainting | ~2GB VRAM |
| 可选修复 | SD 1.5 Inpainting | ~4GB VRAM，高质量纹理 |
| PSD 文字 | ag-psd LayerTextData | 双击可编辑，自动 fauxBold |

## 已知限制

1. PaddlePaddle GPU (CUDA 11.8) 与 torch (CUDA 12.1) 共存时需 `import torch` 先于 `import paddleocr`
2. PP-OCRv5 对英文空格识别不稳定，依赖词表后处理
3. 字体粗细/风格检测为启发式算法，极端字体可能误判
4. SD Inpainting 需 4GB+ VRAM，大图自动缩放到 768px
5. PSD 在 Photoshop 打开时有"需要更新图层"提示，确认后正常

## 性能参考

| 指标 | v2.1 (CPU) | v2.2 (GPU) |
|------|-----------|-----------|
| OCR 每张 | ~0.86s | **~0.38s** |
| 总耗时 (12张) | ~25min | ~18min |
| 峰值显存 | ~3GB | ~4.5GB |
| OCR 置信度 | 0.992 (PP-OCRv4) | 0.958 (PP-OCRv5) |

GPU: RTX 3060 (6GB)
