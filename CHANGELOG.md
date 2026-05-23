# 项目开发日志

## 2026-05-24 — v2.1 精度提升版

### 精度改进

| 维度 | v1.0 (EasyOCR) | v2.1 (PaddleOCR) |
|------|---------------|-----------------|
| OCR 引擎 | EasyOCR | PaddleOCR PP-OCRv4 (CPU) |
| 中文置信度 | 0.66-0.95 | 0.992-1.000 |
| 英文识别 | 漏检 | 检出 + 自动拆分合并词 |
| 文字颜色 | K-means 全域采样 (偏差大) | Otsu 二值化+前景采样 |
| 字体粗细 | 无 | Distance Transform 检测 |
| 字体风格 | 无 | 水平投影方差检测 |
| PSD 字体 | 统一 SimHei | Microsoft YaHei / FangSong / SimSun |
| 背景修复 | LaMa only | LaMa + SD 1.5 Inpainting 可选 |
| 低置信过滤 | 无 | <50% 自动丢弃 |
| VRAM 管理 | 单次加载 | SAM→SD 两阶段分离显存 |

### 目录结构重组

```
v1.0:  pipeline_output/ 混杂源码和产物
v2.1:  源码在根目录, 产物在 output/
```

### 依赖变更

```bash
# 新增
pip install paddlepaddle paddleocr==2.8.1
pip install diffusers  # SD Inpainting 可选

# 移除
# EasyOCR 不再使用
# paddlepaddle-gpu → paddlepaddle (CPU, 避免与 torch CUDA 冲突)
```

---

## 2026-05-23 — v1.0 初始版本

### 背景

电商详情页图片 (12 张 WEBP) → 分层 PSD，以便在 Photoshop 中编辑。

### 技术演进

#### v0.1 — SAM 直接分割 + psd-tools 导出 (已废弃)
- SAM 原图分辨率全图层分割
- psd-tools 写入 PSD
- **问题**: GPU 6GB OOM, 不支持中文图层名

#### v0.2 — 缩放 + SAM + ASCII 图层名 (部分成功)
- 缩放到 1024px SAM 分割
- ASCII 图层名绕过编码限制
- **结果**: 12 PSD 生成成功, 但文字不可编辑, 边缘有锯齿

#### v1.0 — 完整 4 步管线

| 步骤 | 技术选型 | 产出 |
|------|----------|------|
| Step 1: 版面分析 | EasyOCR + SAM | 图层元数据 |
| Step 2: 图层剥离 | LaMa Inpainting + Crop | 干净背景 + 前景 PNG |
| Step 3: 中间状态 | Python JSON | 结构化描述文件 |
| Step 4: PSD 编译 | Node.js + ag-psd | 可编辑文字图层 PSD |

### 关键问题解决

| 问题 | 解决 |
|------|------|
| OpenCV 中文路径 | `np.frombuffer + cv2.imdecode` |
| GPU OOM (6GB) | 缩放到 1024px 做 SAM, mask 映射回原图 |
| psd-tools MacRoman 编码 | 改用 ag-psd (Node.js) |
| 文字图层栅格化 | ag-psd LayerTextData |
| SAM 文字 mask 与 OCR 冲突 | SAM 文字型 mask 不生成图层 |
