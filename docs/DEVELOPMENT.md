# FinBERT 开发文档

## 项目架构

```
finbert/
├── app/                    # 应用核心代码
│   ├── __init__.py
│   ├── config.py          # 配置文件
│   ├── main.py            # FastAPI 入口
│   ├── model.py           # 模型加载和推理
│   └── schemas.py         # Pydantic 数据模型
├── docs/                   # 文档
│   ├── API.md             # API 接口文档
│   └── DEVELOPMENT.md     # 开发文档
├── scripts/                # 脚本工具
│   ├── download_model.py  # 模型下载
│   └── test_tnews.py      # TNEWS 数据集测试
├── k8s/                    # Kubernetes 部署配置
│   └── deployment.yaml    # K8s 部署文件
├── Dockerfile             # Docker 镜像构建
├── docker-compose.yml     # Docker Compose 配置
├── requirements.txt       # Python 依赖
└── README.md              # 项目说明
```

---

## 环境配置

### 系统要求

- Python 3.10+
- CUDA 11.8+ (GPU 推理)
- Docker 20.10+ (容器化部署)

### 本地开发环境

```bash
# 创建 conda 环境
conda create -n finbert python=3.10 -y
conda activate finbert

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| MODEL_NAME | hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2 | 模型名称或本地路径 |
| DEVICE | cuda (如可用) | 推理设备 |
| API_HOST | 0.0.0.0 | API 监听地址 |
| API_PORT | 8888 | API 监听端口 |
| HF_ENDPOINT | https://hf-mirror.com | HuggingFace 镜像源 |
| TRANSFORMERS_OFFLINE | 0 | 离线模式 |

---

## 模型说明

### 使用的模型

**hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2**

- 基于 BERT-base-chinese 微调
- 专为中文金融新闻情绪分析优化
- 三分类：Positive（正面）、Negative（负面）、Neutral（中性）

### 模型下载

```bash
# 使用镜像源下载
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py

# 或手动下载到本地
from huggingface_hub import snapshot_download
snapshot_download(
    'hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2',
    local_dir='./models/finbert'
)
```

---

## 开发指南

### 启动开发服务器

```bash
# 直接启动
uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload

# 或使用脚本
python -m uvicorn app.main:app --reload
```

### 代码结构说明

#### config.py
配置管理，包括模型路径、设备选择、情绪标签映射等。

#### model.py
模型加载和推理核心逻辑。

#### main.py
FastAPI 应用入口，定义 API 路由。

#### schemas.py
Pydantic 数据模型，定义请求和响应格式。

---

## 部署

### Docker 部署（推荐）

```bash
# 构建镜像
docker build -t finbert-sentiment:latest .

# 运行容器（GPU）
docker run -d --name finbert \
  --gpus all \
  -p 8888:8888 \
  -v /path/to/models:/models \
  -e MODEL_NAME=/models \
  -e DEVICE=cuda \
  finbert-sentiment:latest

# 运行容器（CPU）
docker run -d --name finbert \
  -p 8888:8888 \
  finbert-sentiment:latest
```

### Docker Compose 部署

```bash
docker-compose up -d
```

### Kubernetes 部署

```bash
# 部署到 K8s
kubectl apply -f k8s/deployment.yaml
```

---

## 常见问题

### Q: 模型下载失败？

使用国内镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: CUDA out of memory？

- 减小批处理大小
- 使用 CPU 推理
- 检查是否有其他进程占用 GPU

### Q: 离线环境如何使用？

1. 先在有网络的环境下载模型到本地
2. 设置环境变量使用本地路径：
```bash
export MODEL_NAME=/path/to/local/model
export TRANSFORMERS_OFFLINE=1
```
