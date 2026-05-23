# JPG→PSD 智能分层管线

将电商详情页扁平图片（WEBP/JPG/PNG）自动拆解为分层 PSD，支持可编辑文字图层。

## 管线架构

```
原始图片 (WEBP/JPG/PNG)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Stage 1: 版面分析与目标检测                     │
│  · PaddleOCR PP-OCRv4 → 文本/坐标/颜色/字体属性  │
│  · SAM 目标分割 → 产品图/装饰元素 mask           │
│  产出: 图层元数据 + 裁剪 PNG + Inpaint mask      │
├──────────────────────────────────────────────┤
│ Stage 2: 背景修复 + JSON 编译                   │
│  · LaMa / SD Inpainting → 干净背景              │
│  · 中间 JSON 描述 (含置信度/字体元数据)           │
│  产出: JSON + 干净背景 PNG                      │
├──────────────────────────────────────────────┤
│ Step 4: PSD 二进制编译 (compile_psd.js)         │
│  · ag-psd 写入 → 可编辑文字图层                  │
│  · 低置信文字自动丢弃                            │
│  产出: .psd 文件                                │
└──────────────────────────────────────────────┘
```

## 精度特性 (v2.1)

| 特性 | 技术 | 说明 |
|------|------|------|
| 中文 OCR | PaddleOCR PP-OCRv4 (CPU) | 置信度 0.99+，比 EasyOCR 提升 15-20% |
| 英文 OCR | PP-OCRv4 + 词表拆分 | 自动修复 "PRODUCTINFORMATION" → "PRODUCT INFORMATION" |
| 文字颜色 | Otsu 二值化 + 前景 K-means | 精确提取文字色，不受背景色干扰 |
| 字体粗细 | Distance Transform | 检测 thin / normal / bold / extra-bold |
| 字体风格 | 水平投影方差 | 检测 sans-serif / serif |
| 字体映射 | 粗细+风格→PS字体名 | Microsoft YaHei / FangSong / SimSun |
| 背景修复 (默认) | LaMa Inpainting | 快速，低 VRAM (~2GB) |
| 背景修复 (可选) | SD 1.5 Inpainting | 高质量纹理修复，需 ~4GB VRAM |
| PSD 文字图层 | ag-psd LayerTextData | 双击可编辑，自动 fauxBold |

## 目录结构

```
topsd/
├── pipeline.py              # Stage 1-2 主脚本 (Python)
├── pipeline_output/
│   ├── compile_psd.js       # PSD 编译器 (Node.js)
│   ├── temp/                # 中间产物 (背景 + 图层 PNG)
│   ├── json/                # JSON 图层描述
│   ├── psd/                 # 最终 PSD
│   ├── package.json         # Node 依赖
│   └── node_modules/        # ag-psd, pngjs
├── decompose_to_psd.py      # v1 旧版 (SAM only, 仅供参考)
└── *.webp                   # 原始图片 (用户提供)
```

## 依赖环境

### Python
| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.5.1+cu121 | 深度学习后端 |
| paddlepaddle | 3.3.1 (CPU) | PaddleOCR 引擎 |
| paddleocr | 2.8.1 | 中文文字检测识别 |
| segment-anything | - | SAM 目标分割 |
| simple-lama-inpainting | 0.1.2 | LaMa 背景修复 |
| diffusers | 0.38.0 | SD Inpainting (可选) |
| opencv-python | - | 图像处理 |
| Pillow | - | 图像读写 |
| numpy | - | 数组运算 |

### Node.js
| 包 | 用途 |
|----|------|
| ag-psd | PSD 二进制读写 (支持文字图层) |
| pngjs | PNG 像素数据解析 |

### 模型文件
| 模型 | 大小 | 路径 | 说明 |
|------|------|------|------|
| sam_vit_b_01ec64.pth | 358MB | ComfyUI/models/sams/ | 需手动下载 |
| PP-OCRv4 det/rec/cls | ~18MB | ~/.paddleocr/whl/ | 自动下载 |
| big-lama.pt | 196MB | ~/.cache/torch/hub/ | 自动下载 |
| SD 1.5 Inpainting | ~5GB | ~/.cache/huggingface/ | 首次运行自动下载 |

## 使用方式

### 1. 安装依赖

```bash
# Python
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle paddleocr==2.8.1
pip install segment-anything simple-lama-inpainting opencv-python Pillow numpy
pip install diffusers  # 可选: SD Inpainting

# Node.js
cd pipeline_output
npm install
```

### 2. 运行管线

```bash
# 默认 LaMa 模式 (推荐, 省显存)
cd topsd
python pipeline.py

# 切换到 SD 模式 (编辑 pipeline.py 顶部的 INPAINT_MODE = "sd")

# 编译 PSD
cd pipeline_output
node compile_psd.js
```

### 3. 仅编译 PSD (JSON 已生成时)

```bash
cd pipeline_output
node compile_psd.js
```

## JSON 描述格式 (v2.1)

```json
{
  "sourceImage": "detail.webp",
  "width": 2480,
  "height": 2132,
  "ocrEngine": "PaddleOCR PP-OCRv4",
  "inpaintMode": "lama",
  "layers": [
    {
      "type": "background",
      "name": "背景",
      "imagePath": "./temp/detail_clean_bg.png"
    },
    {
      "type": "image",
      "name": "product_2",
      "label": "product",
      "imagePath": "./temp/detail_product_2.png",
      "left": 0, "top": 7,
      "width": 2480, "height": 2118,
      "stability": 0.939
    },
    {
      "type": "text",
      "name": "文字_产品信息",
      "text": "产品信息",
      "left": 913, "top": 632,
      "width": 666, "height": 155,
      "fontSize": 155,
      "color": "#23394e",
      "confidence": 0.997,
      "fontWeight": "thin",
      "fontStyle": "sans-serif",
      "fontName": "Microsoft YaHei"
    }
  ]
}
```

## 可调参数

### pipeline.py

```python
INPAINT_MODE = "lama"       # "lama" (快) 或 "sd" (高质量)
OCR_CONF_THRESHOLD = 0.50   # OCR 最低置信度
SAM_MAX_SIDE = 1024         # SAM 处理分辨率 (越大越精细)
SAM_PARAMS = {
    "points_per_side": 32,        # 采样密度 (16→快, 64→精细)
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.92,
}
```

### compile_psd.js

```javascript
TEXT_CONFIDENCE_THRESHOLD = 0.50  // 低置信文字丢弃阈值
```

## 已知限制

1. **文字图层**: ag-psd 在 Photoshop 打开时有"需要更新图层"提示，点击确认后正常可编辑
2. **字体**: 粗细/风格检测为启发式算法，极端字体可能误判
3. **英文**: PP-OCRv4 对英文空格识别不稳定，依赖词表后处理
4. **SD Inpainting**: 需 4GB+ VRAM，大图自动缩放到 768px 处理
5. **PaddleOCR**: CPU 模式约 0.9s/张，大批量可接受

## 性能参考

- GPU: RTX 3060 (6GB)
- LaMa 模式: ~2-3 分钟/张 (OCR 0.9s + SAM ~30s + LaMa ~60s)
- SD 模式: ~3-5 分钟/张 (OCR 0.9s + SAM ~30s + SD ~120s)
- 峰值显存 (LaMa): ~3GB
- 峰值显存 (SD): ~5GB
- 12 张图片 LaMa 总耗时: ~25 分钟
