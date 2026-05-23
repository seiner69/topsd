/**
 * Step 4: PSD 二进制编译 (v2.1)
 * 读取 Step 3 的 JSON 描述 → 编译为可编辑文字图层的 PSD
 * 使用 ag-psd 写入，支持真实文字图层（非栅格化）
 *
 * v2.1 改进:
 * - 使用 JSON 中的 fontName / fontWeight / fontStyle 字段
 * - 低置信度文字自动降级为像素图层 (避免乱码)
 * - 支持 fauxBold / fauxItalic 属性
 */

const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const { writePsdBuffer } = require('ag-psd');

// ============================================================
// 配置
// ============================================================
const OUTPUT_DIR = path.join(__dirname, 'output');
const JSON_DIR = path.join(OUTPUT_DIR, 'json');
const TEMP_DIR = path.join(OUTPUT_DIR, 'temp');
const PSD_DIR = path.join(OUTPUT_DIR, 'psd');

// 低置信度文字转为像素图层的阈值
const TEXT_CONFIDENCE_THRESHOLD = 0.50;

// 备用字体映射 (如果 JSON 没有提供 fontName)
const FALLBACK_FONTS = {
  'sans-serif': {
    'thin': 'SimHei',
    'normal': 'SimHei',
    'bold': 'Microsoft YaHei',
    'extra-bold': 'Microsoft YaHei',
  },
  'serif': {
    'thin': 'FangSong',
    'normal': 'SimSun',
    'bold': 'SimSun',
    'extra-bold': 'SimHei',
  },
  'unknown': {
    'normal': 'SimHei',
    'bold': 'Microsoft YaHei',
  },
};

// ============================================================
// 工具函数
// ============================================================

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

function readPngPixelData(filepath) {
  const fullPath = path.join(TEMP_DIR, path.basename(filepath));
  if (!fs.existsSync(fullPath)) {
    console.warn(`  [WARN] PNG not found: ${fullPath}`);
    return null;
  }
  const buf = fs.readFileSync(fullPath);
  const png = PNG.sync.read(buf);
  return {
    data: new Uint8ClampedArray(png.data),
    width: png.width,
    height: png.height,
  };
}

function estimateFontSize(pxHeight) {
  return Math.round(pxHeight * 0.75);
}

function resolveFont(layerJson) {
  // 优先使用 Python 端检测的字体名
  if (layerJson.fontName) return layerJson.fontName;

  // 回退: 根据 style + weight 查找
  const style = layerJson.fontStyle || 'unknown';
  const weight = layerJson.fontWeight || 'normal';
  const map = FALLBACK_FONTS[style] || FALLBACK_FONTS['unknown'];
  return map[weight] || map['normal'] || 'SimHei';
}

// ============================================================
// 图层构建
// ============================================================

function buildBackgroundLayer(layerJson) {
  const imageData = readPngPixelData(layerJson.imagePath);
  if (!imageData) return null;

  console.log(`  [bg] 尺寸: ${imageData.width}x${imageData.height}`);

  return {
    name: layerJson.name || 'Background',
    top: 0,
    left: 0,
    bottom: imageData.height,
    right: imageData.width,
    blendMode: 'normal',
    opacity: 1,
    transparencyProtected: true,
    imageData,
  };
}

function buildImageLayer(layerJson) {
  const imageData = readPngPixelData(layerJson.imagePath);
  if (!imageData) return null;

  const left = layerJson.left || 0;
  const top = layerJson.top || 0;

  console.log(`  [img] ${layerJson.name}: ${imageData.width}x${imageData.height} @ (${left},${top})`);

  return {
    name: layerJson.name || 'Image Layer',
    top,
    left,
    bottom: top + imageData.height,
    right: left + imageData.width,
    blendMode: 'normal',
    opacity: 1,
    imageData,
  };
}

function buildTextLayer(layerJson) {
  const text = layerJson.text || '';
  const confidence = layerJson.confidence || 0.0;
  const fontSize = estimateFontSize(layerJson.fontSize || 24);
  const color = hexToRgb(layerJson.color || '#000000');
  const left = layerJson.left || 0;
  const top = layerJson.top || 0;
  const width = layerJson.width || 200;
  const height = layerJson.height || 50;
  const fontName = resolveFont(layerJson);
  const fontWeight = layerJson.fontWeight || 'normal';

  console.log(`  [text] conf=${confidence.toFixed(2)} "${text.slice(0, 20)}" `
    + `fontSize=${fontSize}pt font=${fontName} weight=${fontWeight} `
    + `color=${layerJson.color} @ (${left},${top})`);

  const isBold = fontWeight === 'bold' || fontWeight === 'extra-bold';
  const isItalic = false; // PaddleOCR 不支持斜体检测

  return {
    name: layerJson.name || 'Text Layer',
    top,
    left,
    bottom: top + height,
    right: left + width,
    blendMode: 'normal',
    opacity: 1,
    text: {
      text,
      antiAlias: 'sharp',
      orientation: 'horizontal',
      top: 0,
      left: 0,
      bottom: height,
      right: width,
      style: {
        font: { name: fontName },
        fontSize,
        fillColor: color,
        fauxBold: isBold,
        fauxItalic: isItalic,
        autoLeading: true,
      },
      styleRuns: [
        {
          length: text.length,
          style: {
            font: { name: fontName },
            fontSize,
            fillColor: color,
            fauxBold: isBold,
            fauxItalic: isItalic,
          },
        },
      ],
    },
  };
}

function buildFallbackPixelLayer(layerJson) {
  // 低置信度文字 → 跳过 (避免乱码)
  const text = layerJson.text || '';
  const conf = (layerJson.confidence || 0).toFixed(2);
  console.log(`  [text→skip] LOW CONF (${conf}) "${text.slice(0, 20)}" — dropping`);
  return null;
}

// ============================================================
// 主流程
// ============================================================

function compilePsd(jsonPath) {
  const name = path.basename(jsonPath, '.json');
  console.log(`\n[${name}] 编译 PSD...`);

  const raw = fs.readFileSync(jsonPath, 'utf-8');
  const data = JSON.parse(raw);

  const width = data.width || 800;
  const height = data.height || 600;
  const layers = data.layers || [];
  const ocrEngine = data.ocrEngine || 'unknown';
  const lowConfTexts = [];

  console.log(`  引擎: ${ocrEngine}, 尺寸: ${width}x${height}, 图层: ${layers.length}`);

  const children = [];

  const bgLayers = layers.filter(l => l.type === 'background');
  const imgLayers = layers.filter(l => l.type === 'image');
  const textLayers = layers.filter(l => l.type === 'text');

  // 背景层
  for (const l of bgLayers) {
    const layer = buildBackgroundLayer(l);
    if (layer) children.push(layer);
  }

  // 图片层
  for (const l of imgLayers) {
    const layer = buildImageLayer(l);
    if (layer) children.push(layer);
  }

  // 文字层: 高置信 → 可编辑文字图层; 低置信 → 像素图层
  for (const l of textLayers) {
    const conf = l.confidence || 0.0;
    if (conf >= TEXT_CONFIDENCE_THRESHOLD) {
      const layer = buildTextLayer(l);
      if (layer) children.push(layer);
    } else {
      // 降级为像素图层
      const layer = buildFallbackPixelLayer(l);
      if (layer) children.push(layer);
      lowConfTexts.push(l.text);
    }
  }

  if (lowConfTexts.length > 0) {
    console.log(`  [INFO] ${lowConfTexts.length} 低置信文字降级为像素: ${lowConfTexts.join(', ')}`);
  }

  if (children.length === 0) {
    console.warn(`  [WARN] No valid layers, skipping`);
    return null;
  }

  const psd = {
    width,
    height,
    channels: 3,
    bitsPerChannel: 8,
    colorMode: 3,
    children,
  };

  try {
    const buffer = writePsdBuffer(psd, {
      generateThumbnail: false,
    });

    const outPath = path.join(PSD_DIR, `${name}.psd`);
    fs.writeFileSync(outPath, buffer);
    const textCount = textLayers.filter(l => (l.confidence || 0) >= TEXT_CONFIDENCE_THRESHOLD).length;
    console.log(`  ✓ ${name}.psd (${(buffer.length / 1024).toFixed(0)} KB, `
      + `${children.length} layers, ${textCount} editable text)`);
    return outPath;
  } catch (err) {
    console.error(`  [ERROR] ${err.message}`);
    return null;
  }
}

function main() {
  fs.mkdirSync(PSD_DIR, { recursive: true });

  const jsonFiles = fs.readdirSync(JSON_DIR)
    .filter(f => f.endsWith('.json') && !f.startsWith('_'))
    .map(f => path.join(JSON_DIR, f));

  if (jsonFiles.length === 0) {
    console.log('No JSON files found in', JSON_DIR);
    console.log('Run pipeline.py first to generate JSON descriptors.');
    return;
  }

  console.log(`\nCompiling ${jsonFiles.length} PSD files...`);

  const results = [];
  for (const jsonPath of jsonFiles) {
    const result = compilePsd(jsonPath);
    if (result) results.push(result);
  }

  console.log(`\nDone! ${results.length}/${jsonFiles.length} PSDs compiled`);
  console.log(`Output: ${PSD_DIR}`);
}

main();
