# FinBERT 金融情绪分析 API 文档

## 概述

FinBERT 金融情绪分析服务提供 RESTful API，用于分析中文金融新闻文本的情绪倾向（正面/负面/中性）。

**基础 URL**: `http://localhost:8888`

---

## 接口列表

### 1. 健康检查

检查服务状态和模型加载情况。

**请求**
```
GET /health
```

**响应**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态，"ok" 表示正常 |
| model_loaded | boolean | 模型是否已加载 |
| device | string | 推理设备，"cuda" 或 "cpu" |

---

### 2. 单条文本情绪分析

分析单条金融新闻文本的情绪。

**请求**
```
POST /predict
Content-Type: application/json
```

**请求体**
```json
{
  "text": "股市大涨，投资者信心增强"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 是 | 待分析的金融新闻文本 |

**响应**
```json
{
  "text": "股市大涨，投资者信心增强",
  "sentiment": "Positive",
  "sentiment_zh": "正面",
  "confidence": 0.9998,
  "probabilities": {
    "Negative": 0.00003,
    "Neutral": 0.00014,
    "Positive": 0.9998
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 原始输入文本 |
| sentiment | string | 情绪标签（英文）：Positive/Negative/Neutral |
| sentiment_zh | string | 情绪标签（中文）：正面/负面/中性 |
| confidence | float | 置信度（0-1） |
| probabilities | object | 各情绪类别的概率分布 |

---

### 3. 批量文本情绪分析

批量分析多条金融新闻文本。

**请求**
```
POST /predict/batch
Content-Type: application/json
```

**请求体**
```json
{
  "texts": [
    "股市大涨，投资者信心增强",
    "公司业绩下滑，股价暴跌",
    "央行维持利率不变"
  ],
  "batch_size": 32
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| texts | array | 是 | 待分析的文本列表 |
| batch_size | int | 否 | 批处理大小，默认32 |

**响应**
```json
{
  "results": [
    {
      "text": "股市大涨，投资者信心增强",
      "sentiment": "Positive",
      "sentiment_zh": "正面",
      "confidence": 0.9998,
      "probabilities": {...}
    },
    {
      "text": "公司业绩下滑，股价暴跌",
      "sentiment": "Negative",
      "sentiment_zh": "负面",
      "confidence": 0.9876,
      "probabilities": {...}
    },
    {
      "text": "央行维持利率不变",
      "sentiment": "Neutral",
      "sentiment_zh": "中性",
      "confidence": 0.8521,
      "probabilities": {...}
    }
  ],
  "count": 3
}
```

---

## 错误响应

当请求失败时，返回以下格式：

```json
{
  "detail": "错误描述信息"
}
```

**HTTP 状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |

---

## 使用示例

### cURL

```bash
# 健康检查
curl http://localhost:8888/health

# 单条分析
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "股市大涨，投资者信心增强"}'

# 批量分析
curl -X POST "http://localhost:8888/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["新闻1", "新闻2", "新闻3"]}'
```

### Python

```python
import requests

# 单条分析
response = requests.post(
    "http://localhost:8888/predict",
    json={"text": "股市大涨，投资者信心增强"}
)
result = response.json()
print(f"情绪: {result['sentiment_zh']}, 置信度: {result['confidence']:.2%}")

# 批量分析
response = requests.post(
    "http://localhost:8888/predict/batch",
    json={"texts": ["新闻1", "新闻2", "新闻3"]}
)
results = response.json()["results"]
for r in results:
    print(f"{r['text'][:20]}... -> {r['sentiment_zh']}")
```

---

## 性能指标

| 指标 | GPU (RTX 3080) | CPU |
|------|----------------|-----|
| 单条推理延迟 | ~10ms | ~50ms |
| 批量吞吐量 | ~500 条/秒 | ~100 条/秒 |
| 模型加载时间 | ~5s | ~10s |
| 显存占用 | ~1.5GB | - |
