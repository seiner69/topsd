# 项目开发日志

## 2026-05-23 — 初始版本

### 背景

用户有一批电商详情页图片（12 张 WEBP，来自淘宝/天猫详情页），需要将扁平的 JPG/WEBP 拆解为分层的 PSD 文件，以便在 Photoshop 中编辑。

### 技术演进

#### v0.1 — SAM 直接分割 + psd-tools 导出 (已废弃)
- 直接用 SAM 在原图分辨率做全图层分割
- psd-tools 写入 PSD
- **问题**: GPU 6GB 显存不足 (原图 2480×4960)，psd-tools 不支持中文图层名
- **结论**: 不可行

#### v0.2 — 缩放 + SAM + ASCII 图层名 (部分成功)
- 图片缩放到 1024px 后做 SAM 分割
- 用 ASCII 图层名绕过 psd-tools 编码限制
- **结果**: 12 个 PSD 生成成功，但所有图层都是像素层（文字不可编辑）
- **问题**: 图层是直接从原图裁切的，边缘有残留/锯齿

#### v1.0 — 完整 4 步管线 (当前版本)

用户提出完整 4 步方案后重构：

| 步骤 | 技术选型 | 产出 |
|------|----------|------|
| Step 1: 版面分析 | EasyOCR + SAM | 图层元数据 (文本/坐标/颜色/字号) |
| Step 2: 图层剥离 | LaMa Inpainting + Crop | 干净背景 + 透明前景 PNG |
| Step 3: 中间状态 | Python JSON | 结构化图层描述文件 |
| Step 4: PSD 编译 | Node.js + ag-psd | 可编辑文字图层的 PSD |

### 依赖安装记录

```bash
# Python
pip install psd-tools         # v0.1 尝试 (后弃用)
pip install simple-lama-inpainting  # LaMa 背景修复
# easyocr, segment-anything, opencv, torch 已预装

# Node.js
cd pipeline_output
npm init -y
npm install ag-psd   # PSD 读写
npm install pngjs    # PNG 像素解析
```

### 遇到的关键问题与解决

| 问题 | 解决 |
|------|------|
| OpenCV 中文路径 | 用 `np.frombuffer + cv2.imdecode` 替代 `cv2.imread` |
| GPU OOM (6GB) | 图片缩放到 1024px 做 SAM，mask 映射回原图 |
| psd-tools MacRoman 编码 | 改用 ag-psd (Node.js)，天然支持 UTF-8 图层名 |
| 文字图层栅格化 | ag-psd 的 `LayerTextData` 支持真实文字图层 |
| GBK 终端输出乱码 | print 内容本身正常，仅 Windows 终端显示问题 |
| SAM 文字 mask 与 OCR 冲突 | SAM 检测到的文字型 mask 不生成图层，交由 OCR 处理 |

### 处理结果统计

- **输入**: 12 张 WEBP 图片 (1.6MP - 12.3MP)
- **输出**: 12 个 PSD 文件 (6MB - 120MB，合计 334MB)
- **每图平均图层**: 10 层 (2-3 背景/产品层 + 7-10 文字层)
- **总处理时间**: ~30 分钟
- **OCR 检测文本块**: ~100 个 (含中文/英文/数字)
- **SAM 检测目标**: ~150 个 (产品/装饰/背景元素)

### 待优化项

- [ ] PaddleOCR 替代 EasyOCR（更高中文精度）
- [ ] Stable Diffusion Inpainting 替代 LaMa（更好的纹理修复）
- [ ] Photoshop COM API 替代 ag-psd（完美文字图层兼容）
- [ ] 字体粗细/样式自动匹配
- [ ] 艺术字体/竖排文字支持
- [ ] 批量并发处理优化
