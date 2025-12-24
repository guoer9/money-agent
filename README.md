# FinBERT 中文金融新闻情绪分析

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/GPU-CUDA%2011.8-brightgreen.svg" alt="CUDA">
</p>

基于预训练 BERT 模型的中文金融新闻情绪识别服务，提供 RESTful API 接口，支持 Docker 和 Kubernetes 部署。

> 🔗 **Money-Agent 项目分支说明**
> - `main` - 项目主分支
> - `vllm` - vLLM 大语言模型推理服务
> - `finbert` - **本分支：金融情绪分析服务**

---

## ✨ 功能特性

- 🎯 **专业模型** - 使用针对中文金融领域微调的 BERT 模型
- ⚡ **高性能** - GPU 加速推理，单条延迟 ~10ms
- 🔌 **RESTful API** - FastAPI 构建，自动生成 OpenAPI 文档
- 🐳 **容器化** - Docker/K8s 一键部署
- 📊 **三分类** - 正面/负面/中性情绪识别

---

## 🚀 快速开始

### Docker 部署（推荐）

```bash
# 1. 下载模型到本地
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2',
                  local_dir='./models/finbert')
"

# 2. 构建镜像
docker build -t finbert-sentiment:latest .

# 3. 运行服务 (GPU)
docker run -d --name finbert --gpus all -p 8888:8888 \
  -v $(pwd)/models/finbert:/models \
  -e MODEL_NAME=/models -e DEVICE=cuda \
  finbert-sentiment:latest

# 4. 测试
curl http://localhost:8888/health
```

### 本地运行

```bash
# 创建环境
conda create -n finbert python=3.10 -y
conda activate finbert

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8888
```

---

## 📖 API 接口

### 健康检查
```bash
curl http://localhost:8888/health
# {"status":"ok","model_loaded":true,"device":"cuda"}
```

### 单条情绪分析
```bash
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "股市大涨，投资者信心增强"}'
```

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

### 批量分析
```bash
curl -X POST "http://localhost:8888/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["利好消息推动股价上涨", "公司业绩大幅下滑", "市场维持震荡格局"]}'
```

📚 **完整 API 文档**: [docs/API.md](docs/API.md)

---

## 🏗️ 项目结构

```
finbert/
├── app/                    # 核心应用代码
│   ├── config.py          # 配置管理
│   ├── main.py            # FastAPI 入口
│   ├── model.py           # 模型推理
│   └── schemas.py         # 数据模型
├── docs/                   # 文档
│   ├── API.md             # API 接口文档
│   └── DEVELOPMENT.md     # 开发文档
├── scripts/                # 工具脚本
│   ├── download_model.py  # 模型下载
│   └── test_tnews.py      # TNEWS 测试
├── k8s/                    # K8s 部署配置
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔧 配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MODEL_NAME` | hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2 | 模型名称或本地路径 |
| `DEVICE` | cuda (如可用) | 推理设备 |
| `API_PORT` | 8888 | 服务端口 |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace 镜像 |

---

## 📊 性能

| 指标 | GPU (RTX 3080) | CPU |
|------|----------------|-----|
| 单条延迟 | ~10ms | ~50ms |
| 吞吐量 | ~500 条/秒 | ~100 条/秒 |
| 显存占用 | ~1.5GB | - |

---

## 📚 相关文档

- [API 接口文档](docs/API.md)
- [开发文档](docs/DEVELOPMENT.md)
- [Swagger UI](http://localhost:8888/docs) (服务启动后)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License
