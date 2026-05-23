# 开发日志

## 2026-05-24 — v2.2.1 模块化重构 + 目录整理

### 代码重构

791 行单体 `pipeline.py` 拆分为 6 个模块：

| 模块 | 职责 |
|------|------|
| `config.py` | 全局配置常量 |
| `image_utils.py` | 图像 I/O 与 mask 运算 |
| `text_engine.py` | PaddleOCR 封装 + 颜色/字体检测 |
| `sam_engine.py` | SAM 分割 + mask 分类去重 |
| `inpaint_engine.py` | LaMa/SD 背景修复 |
| `pipeline.py` | 纯编排层 (214行) |

### 目录整理

- 输入图片统一移至 `input/` 目录
- 删除 `decompose_to_psd.py` (v1 旧版)
- 删除 `PSD输出/` (旧产物目录)
- 更新 `.gitignore` 适配新结构

---

## 2026-05-24 — v2.2 GPU 加速版

### 核心变更

| 维度 | v2.1 | v2.2 |
|------|------|------|
| PaddleOCR | 2.8.1 | **3.4.1** |
| PaddlePaddle | 2.6.2 CPU | **3.2.2 GPU (CUDA 11.8)** |
| OCR 引擎 | PP-OCRv4 | **PP-OCRv5_server** |
| OCR 设备 | CPU (~0.86s/张) | **GPU (~0.38s/张)** |
| 平均置信度 | 0.992 | 0.958 (校准更严格) |

### 技术要点

- PaddlePaddle GPU (CUDA 11.8) 与 torch (CUDA 12.1) 可共存，需 `import torch` 先于 `import paddleocr`
- PaddlePaddle 3.x GPU 仅通过 PaddlePaddle 官方镜像分发，PyPI 上只有 CPU 版
- PP-OCRv5_server 模型首次运行自动下载到 `~/.paddlex/official_models/`
- PaddleOCR 3.x API 变更: `ocr.predict(img)` 返回 `OCRResult` 对象

---

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

### 依赖变更

```bash
# 新增
pip install paddlepaddle paddleocr==2.8.1
pip install diffusers  # SD Inpainting 可选

# 移除
# EasyOCR 不再使用
```

---

## 2026-05-23 — v1.0 初始版本

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
