# 电商详情页 → PSD 图层拆解管线

将电商详情页扁平图片（WEBP/JPG/PNG）自动拆解为分层的 PSD 文件，支持可编辑文字图层。

## 管线架构

```
原始图片 (WEBP)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Step 1: 版面分析与目标检测                      │
│  · EasyOCR 文字检测 → 文本内容/坐标/颜色/字号     │
│  · SAM 目标分割 → 产品图/模特图/装饰元素边界框    │
│  产出: 图层元数据                               │
├──────────────────────────────────────────────┤
│ Step 2: 图层剥离与背景修复                      │
│  · 前景抠图 → 透明 PNG                          │
│  · LaMa Inpainting → 擦除文字和商品的干净背景     │
│  产出: 独立图层 PNG + 无痕背景图                  │
├──────────────────────────────────────────────┤
│ Step 3: 中间 JSON 状态描述                      │
│  · 统一整理所有元素的坐标、类型、样式             │
│  产出: JSON 描述文件                            │
├──────────────────────────────────────────────┤
│ Step 4: PSD 二进制编译                         │
│  · ag-psd 写入 PSD 文件                        │
│  · 像素图层: 嵌入 PNG 像素数据                   │
│  · 文字图层: 真实可编辑文字（非栅格化）            │
│  产出: .psd 文件                                │
└──────────────────────────────────────────────┘
```

## 目录结构

```
详情页/
├── pipeline.py              # Step 1-3 主脚本 (Python)
├── pipeline_output/
│   ├── compile_psd.js       # Step 4 PSD 编译器 (Node.js)
│   ├── temp/                # 中间产物 (干净背景 + 图层 PNG)
│   ├── json/                # JSON 图层描述文件
│   ├── psd/                 # 最终 PSD 文件
│   ├── package.json         # Node.js 依赖
│   └── node_modules/        # ag-psd, pngjs
└── *.webp                   # 原始图片
```

## 依赖环境

### Python
| 包 | 用途 |
|----|------|
| torch | 深度学习后端 |
| easyocr | 中文文字检测与识别 |
| segment-anything | SAM 目标分割 |
| simple-lama-inpainting | LaMa 背景修复 |
| opencv-python | 图像处理 |
| Pillow | 图像读写 |
| numpy | 数组运算 |

### Node.js
| 包 | 用途 |
|----|------|
| ag-psd | PSD 二进制读写（支持文字图层）|
| pngjs | PNG 像素数据解析 |

### 模型文件
| 模型 | 大小 | 路径 |
|------|------|------|
| sam_vit_b_01ec64.pth | 358MB | ComfyUI/models/sams/ |
| big-lama.pt | 196MB | ~/.cache/torch/hub/checkpoints/ |

## 使用方式

### 1. 运行完整管线

```bash
# Step 1-3: Python (OCR + SAM + LaMa + JSON)
cd C:\Users\86191\Desktop\详情页
python pipeline.py

# Step 4: PSD 编译
cd pipeline_output
node compile_psd.js
```

### 2. 仅运行 PSD 编译 (JSON 已生成时)

```bash
cd C:\Users\86191\Desktop\详情页\pipeline_output
node compile_psd.js
```

### 3. 处理其他图片

将新的 WEBP/PNG/JPG 图片放入 `详情页/` 目录，修改 `pipeline.py` 中的 `INPUT_DIR` 路径后运行。

## JSON 描述格式

```json
{
  "sourceImage": "image.webp",
  "width": 2480,
  "height": 2132,
  "layers": [
    {
      "type": "background",
      "name": "背景",
      "imagePath": "./temp/xxx_clean_bg.png"
    },
    {
      "type": "image",
      "name": "product_2",
      "label": "product",
      "imagePath": "./temp/xxx_product_2.png",
      "left": 0, "top": 7,
      "width": 2480, "height": 2118
    },
    {
      "type": "text",
      "name": "文字_产品信息",
      "text": "产品信息",
      "left": 882, "top": 603,
      "width": 722, "height": 208,
      "fontSize": 208,
      "color": "#f3f8fd"
    }
  ]
}
```

## 图层类型说明

| type | 说明 | PSD 中表现 |
|------|------|-----------|
| background | LaMa 修复的无痕背景 | 像素图层（最底层）|
| image | SAM 检测到的产品图/装饰元素 | 带透明通道的像素图层 |
| text | EasyOCR 识别的文字块 | **可编辑文字图层** |

## 可调参数

### pipeline.py

```python
SAM_MAX_SIDE = 1024       # SAM 处理分辨率 (越大越精细，越吃显存)
SAM_PARAMS = {
    "points_per_side": 32,      # 采样密度 (16→快, 64→精细)
    "pred_iou_thresh": 0.88,    # 质量阈值
    "stability_score_thresh": 0.92,
}
MAX_LAYERS = 30            # 单图最大图层数
```

### compile_psd.js

```javascript
const CN_FONT = {
  default: 'SimHei',       // 默认中文字体
};
```

## 已知限制

1. **文字图层**: ag-psd 在 Photoshop 打开时有"需要更新图层"提示，点击确认后正常可编辑
2. **文字朝向**: 仅支持横排文字，竖排文字可能损坏
3. **字体**: 使用默认黑体 (SimHei)，如需匹配原文字体需手动调整
4. **Inpainting**: LaMa 对渐变/纹理复杂背景可能留下痕迹
5. **OCR**: EasyOCR 对极小文字、艺术字体识别率有限

## 性能参考

- GPU: RTX 3060 (6GB)
- 平均处理时间: 每张图约 2-4 分钟
- 峰值显存: ~5GB
- 12 张图片总耗时: ~30 分钟
