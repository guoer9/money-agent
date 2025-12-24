# Qwen3-8B 新闻分类服务 - 完整配置与API指南

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-qwen3--8b--news--classifier-yellow)](https://huggingface.co/guoer9/qwen3-8b-news-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-vllm--branch-blue)](https://github.com/guoer9/money-agent/tree/vllm)

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [服务配置详解](#3-服务配置详解)
4. [API接口详解](#4-api接口详解)
5. [Metrics监控](#5-metrics监控)
6. [Kubernetes部署](#6-kubernetes部署)
7. [使用示例](#7-使用示例)
8. [故障排查](#8-故障排查)

---

## 1. 项目概述

### 1.1 项目简介

本项目是基于 **Qwen3-8B** 微调的中文新闻分类模型服务，在 TNEWS 数据集上达到 **62.4% 准确率**，超越 ERNIE 3.0 Titan (260B) 等 SOTA 模型。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| 模型 | Qwen3-8B + LoRA 微调 |
| 量化 | 8-bit (BitsAndBytes) |
| 显存 | ~10GB (支持 RTX 3080/4090) |
| 并发 | 3 个并发请求 |
| API | OpenAI 兼容格式 |
| 监控 | Prometheus 集成 |
| 部署 | Docker + Kubernetes |

### 1.3 项目结构

```
qwen_vllm/
├── config.py                    # 服务配置文件
├── Dockerfile                   # 容器构建文件
├── docker-compose.yml           # Docker Compose 配置
├── start.sh                     # 本地启动脚本
├── deploy-k8s.sh               # K8s 部署脚本
├── requirements.txt            # Python 依赖
│
├── models/                     # 模型文件
│   └── qwen-news-classifier-merged/  # 微调后的模型 (16GB)
│
├── scripts/
│   ├── deployment/             # 部署相关
│   │   ├── deploy_with_limits.py    # 主服务 (Flask)
│   │   ├── metrics.py              # Metrics 收集
│   │   ├── monitor_metrics.py      # 实时监控
│   │   └── test_api.py             # API 测试
│   ├── training/               # 训练脚本
│   │   ├── train_qwen.py          # 微调训练
│   │   ├── prepare_data.py        # 数据准备
│   │   └── inference_qwen.py      # 推理测试
│   └── utils/                  # 工具脚本
│
├── k8s/                        # Kubernetes 配置
│   ├── base/                   # 基础配置
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ...
│   └── overlays/production/    # 生产环境覆盖
│
└── docs/                       # 文档
```

---

## 2. 环境配置

### 2.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | RTX 3080 10GB | RTX 4090 24GB |
| 内存 | 16GB | 32GB |
| 磁盘 | 30GB | 50GB |
| CPU | 8核 | 16核 |

### 2.2 软件依赖

```bash
# Python 版本
Python >= 3.9

# CUDA 版本
CUDA >= 12.0

# 主要依赖
torch >= 2.0
transformers >= 4.40
bitsandbytes >= 0.43
flask >= 2.0
flask-limiter >= 3.0
gunicorn >= 21.0
```

### 2.3 安装步骤

```bash
# 1. 克隆项目
git clone -b vllm https://github.com/guoer9/money-agent.git
cd money-agent

# 2. 创建虚拟环境
conda create -n vllm-deploy python=3.10
conda activate vllm-deploy

# 3. 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 4. 下载模型 (从 Hugging Face)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('guoer9/qwen3-8b-news-classifier', local_dir='models/qwen-news-classifier-merged')
"

# 5. 启动服务
bash start.sh
```

---

## 3. 服务配置详解

### 3.1 配置文件 (`config.py`)

```python
# ============================================
# 服务配置
# ============================================

# 服务地址
HOST = "0.0.0.0"          # 监听所有网卡
PORT = 8000               # 服务端口

# 模型路径
MODEL_PATH = "./models/qwen-news-classifier-merged"

# ============================================
# 并发控制（基于显存优化）
# ============================================

MAX_CONCURRENT_REQUESTS = 3   # 最大并发请求数
                              # RTX 3080 10GB: 建议 2-3
                              # RTX 4090 24GB: 建议 4-6

REQUEST_QUEUE_SIZE = 10       # 请求队列大小
                              # 超过后返回 503

# ============================================
# 速率限制
# ============================================

RATE_LIMIT_PER_MINUTE = 10    # 每IP每分钟请求数
RATE_LIMIT_PER_HOUR = 100     # 每IP每小时请求数

# ============================================
# 推理参数
# ============================================

MAX_TOKENS = 512              # 最大生成token数
DEFAULT_TEMPERATURE = 0.3     # 默认温度 (越低越确定)
TIMEOUT_SECONDS = 120         # 请求超时时间

# ============================================
# Gunicorn 配置
# ============================================

WORKERS = 1                   # Worker数量 (GPU模型建议1)
THREADS = 4                   # 每个worker的线程数
```

### 3.2 配置说明

#### 并发控制原理

```
请求 → [队列] → [信号量控制] → [GPU推理] → 响应
         ↓           ↓
      最大10个    最大3个并发
```

- **REQUEST_QUEUE_SIZE**: 等待队列大小，超过返回 503
- **MAX_CONCURRENT_REQUESTS**: 同时进行 GPU 推理的请求数
- **信号量机制**: 使用 `threading.Semaphore` 控制 GPU 访问

#### 显存与并发的关系

| GPU | 显存 | 建议并发 | 说明 |
|-----|------|----------|------|
| RTX 3080 | 10GB | 2-3 | 8-bit量化后约7GB |
| RTX 3090 | 24GB | 4-6 | 余量充足 |
| RTX 4090 | 24GB | 5-8 | 推理更快 |
| RTX 5090 | 32GB | 8-10 | 高性能 |

### 3.3 量化配置

```python
# 8-bit 量化配置 (deploy_with_limits.py)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,                    # 启用8-bit量化
    llm_int8_threshold=6.0,               # 异常值阈值
    llm_int8_has_fp16_weight=False,       # 权重格式
    llm_int8_enable_fp32_cpu_offload=True # CPU offload
)
```

**量化效果**:
- 原始模型: ~16GB
- 8-bit量化后: ~8GB
- 推理显存: ~10GB (含KV cache)

---

## 4. API接口详解

### 4.1 接口总览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话补全 (推荐) |
| `/v1/completions` | POST | 文本补全 |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 统计信息 |
| `/metrics` | GET | Metrics (JSON) |
| `/metrics/prometheus` | GET | Metrics (Prometheus) |

### 4.2 对话补全接口 (推荐)

**端点**: `POST /v1/chat/completions`

**请求格式**:
```json
{
    "messages": [
        {"role": "system", "content": "你是一个新闻分类助手"},
        {"role": "user", "content": "请分类这条新闻：央行宣布降息25个基点"}
    ],
    "max_tokens": 100,
    "temperature": 0.3
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| messages | array | ✅ | - | 对话消息列表 |
| max_tokens | int | ❌ | 100 | 最大生成token数 (上限512) |
| temperature | float | ❌ | 0.7 | 温度 (0-1, 越低越确定) |

**messages 格式**:
```json
[
    {"role": "system", "content": "系统提示"},
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "助手回复"},
    {"role": "user", "content": "后续问题"}
]
```

**响应格式**:
```json
{
    "id": "chatcmpl-123456789",
    "object": "chat.completion",
    "created": 1703404800,
    "model": "qwen-news-classifier",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "这条新闻属于【财经】类别。\n\n原因：新闻内容涉及央行货币政策..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 80,
        "total_tokens": 130
    }
}
```

### 4.3 文本补全接口

**端点**: `POST /v1/completions`

**请求格式**:
```json
{
    "prompt": "请分类以下新闻：苹果公司发布新款iPhone，股价上涨5%。\n类别：",
    "max_tokens": 50,
    "temperature": 0.3
}
```

**响应格式**:
```json
{
    "id": "cmpl-123456789",
    "object": "text_completion",
    "created": 1703404800,
    "model": "qwen-news-classifier",
    "choices": [
        {
            "text": "科技/财经",
            "index": 0,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35
    }
}
```

### 4.4 健康检查接口

**端点**: `GET /health`

**响应**:
```json
{
    "status": "ok",
    "gpu": {
        "allocated_gb": 7.52,
        "reserved_gb": 8.0,
        "free_gb": 2.48
    },
    "limits": {
        "max_concurrent_requests": 3,
        "max_queue_size": 10,
        "rate_limit": "10 requests/minute per IP"
    }
}
```

### 4.5 统计信息接口

**端点**: `GET /stats`

**响应**:
```json
{
    "statistics": {
        "total_requests": 1250,
        "successful_requests": 1200,
        "failed_requests": 50,
        "queue_full_count": 15,
        "current_queue_size": 2,
        "peak_queue_size": 8
    },
    "current_queue_size": 2,
    "available_slots": 1
}
```

### 4.6 模型列表接口

**端点**: `GET /v1/models`

**响应**:
```json
{
    "object": "list",
    "data": [
        {
            "id": "qwen-news-classifier",
            "object": "model",
            "created": 0,
            "owned_by": "user"
        }
    ]
}
```

### 4.7 错误响应

**速率限制 (429)**:
```json
{
    "error": "Rate limit exceeded"
}
```

**队列已满 (503)**:
```json
{
    "error": "服务繁忙，请稍后重试",
    "queue_size": 10,
    "max_queue_size": 10
}
```

**内部错误 (500)**:
```json
{
    "error": "CUDA out of memory..."
}
```

---

## 5. Metrics监控

### 5.1 核心指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `num_waiting_requests` | Gauge | 等待队列中的请求数 |
| `num_running_requests` | Gauge | 正在处理的请求数 |
| `total_requests` | Counter | 总请求数 |
| `ttft_mean` | Gauge | 平均首Token延迟 (秒) |
| `ttft_p50/p95/p99` | Gauge | TTFT 百分位数 |
| `decoding_throughput_mean` | Gauge | 平均解码吞吐量 (tokens/秒) |
| `total_throughput` | Gauge | 总吞吐量 (tokens/秒) |

### 5.2 Metrics 接口

#### JSON 格式

**端点**: `GET /metrics`

```json
{
    "num_waiting_requests": 0,
    "num_running_requests": 1,
    "total_requests": 150,
    "total_tokens_generated": 12500,
    "slo": {
        "ttft_mean": 0.45,
        "ttft_p50": 0.42,
        "ttft_p95": 0.68,
        "ttft_p99": 0.85,
        "decoding_throughput_mean": 48.5,
        "decoding_throughput_p50": 50.2,
        "decoding_throughput_p95": 42.1,
        "total_throughput": 45.3,
        "sample_size": 100
    }
}
```

#### Prometheus 格式

**端点**: `GET /metrics/prometheus`

```
# HELP vllm_num_waiting_requests Number of requests waiting in queue
# TYPE vllm_num_waiting_requests gauge
vllm_num_waiting_requests 0

# HELP vllm_num_running_requests Number of requests currently running
# TYPE vllm_num_running_requests gauge
vllm_num_running_requests 1

# HELP vllm_ttft_p95 P95 time to first token in seconds
# TYPE vllm_ttft_p95 gauge
vllm_ttft_p95 0.68

# HELP vllm_decoding_throughput_mean Mean decoding throughput in tokens/sec
# TYPE vllm_decoding_throughput_mean gauge
vllm_decoding_throughput_mean 48.5
```

### 5.3 实时监控

```bash
# 使用监控脚本
python scripts/deployment/monitor_metrics.py

# 输出示例:
# ========================================
# vLLM Metrics Monitor (每5秒刷新)
# ========================================
# 
# 📊 当前状态:
#   等待请求: 0
#   运行请求: 1
#   总请求数: 150
# 
# ⏱️ SLO指标:
#   TTFT P50: 0.42s
#   TTFT P95: 0.68s
#   吞吐量: 48.5 tokens/s
```

### 5.4 Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'
    scrape_interval: 15s
```

### 5.5 告警规则

```yaml
# 推荐告警规则
groups:
  - name: vllm
    rules:
      - alert: HighQueueLength
        expr: vllm_num_waiting_requests > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "请求队列过长"
          
      - alert: HighTTFT
        expr: vllm_ttft_p95 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT延迟过高"
```

---

## 6. Kubernetes部署

### 6.1 快速部署

```bash
# 1. 构建镜像
docker build -t qwen-vllm:latest .

# 2. 部署到K8s
kubectl apply -k k8s/base

# 3. 查看状态
kubectl get pods -l app=qwen-vllm

# 4. 访问服务
kubectl port-forward svc/qwen-vllm 8000:8000
```

### 6.2 Deployment 配置

```yaml
# k8s/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-vllm
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: qwen-vllm
        image: docker.io/library/qwen-vllm:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        env:
        - name: VLLM_MODEL
          value: "/app/models/qwen-news-classifier-merged"
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        hostPath:
          path: /home/zch/qwen_vllm/models/qwen-news-classifier-merged
```

### 6.3 Service 配置

```yaml
# k8s/base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: qwen-vllm
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: qwen-vllm
```

---

## 7. 使用示例

### 7.1 Python 调用

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

# 新闻分类
response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "请分类这条新闻：央行宣布降息25个基点，市场反应积极"}
    ],
    "max_tokens": 100,
    "temperature": 0.3
})

result = response.json()
print(result['choices'][0]['message']['content'])
# 输出: 这条新闻属于【财经】类别...
```

### 7.2 cURL 调用

```bash
# 新闻分类
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "请分类：特斯拉股价大涨10%"}
    ],
    "max_tokens": 100
  }'

# 健康检查
curl http://localhost:8000/health

# 查看统计
curl http://localhost:8000/stats

# Prometheus指标
curl http://localhost:8000/metrics/prometheus
```

### 7.3 批量处理

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def classify_news(news):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": f"分类：{news}"}],
            "max_tokens": 50,
            "temperature": 0.3
        },
        timeout=30
    )
    return response.json()['choices'][0]['message']['content']

news_list = [
    "央行降息25个基点",
    "苹果发布新款iPhone",
    "国足亚洲杯出局",
    # ...
]

# 并发处理 (建议不超过3个并发)
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(classify_news, news_list))
```

### 7.4 流式响应 (开发中)

```python
# 注意: 当前版本暂不支持流式响应
# 如需流式响应，建议使用 vLLM 官方引擎
```

---

## 8. 故障排查

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CUDA out of memory | 显存不足 | 降低 `MAX_CONCURRENT_REQUESTS` |
| 503 服务繁忙 | 队列已满 | 等待或增加队列大小 |
| 429 速率限制 | 请求过快 | 降低请求频率 |
| 连接超时 | 推理时间长 | 增加超时时间 |
| 模型加载失败 | 路径错误 | 检查 `MODEL_PATH` |

### 8.2 日志查看

```bash
# 本地服务
tail -f logs/vllm.log

# K8s Pod
kubectl logs -f deployment/qwen-vllm

# 实时GPU状态
watch -n 1 nvidia-smi
```

### 8.3 性能调优

```python
# config.py 调优建议

# 高吞吐场景
MAX_CONCURRENT_REQUESTS = 5
REQUEST_QUEUE_SIZE = 20
MAX_TOKENS = 256  # 减少最大token

# 低延迟场景
MAX_CONCURRENT_REQUESTS = 2
REQUEST_QUEUE_SIZE = 5
DEFAULT_TEMPERATURE = 0.1  # 更确定的输出
```

### 8.4 健康检查

```bash
# 完整健康检查脚本
#!/bin/bash
echo "=== 健康检查 ==="

# 1. 服务状态
curl -s http://localhost:8000/health | jq

# 2. GPU状态
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# 3. 进程状态
ps aux | grep gunicorn

# 4. 队列状态
curl -s http://localhost:8000/stats | jq
```

---

## 附录

### A. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_MODEL` | 模型路径 | `./models/qwen-news-classifier-merged` |
| `VLLM_PORT` | 服务端口 | `8000` |
| `CUDA_VISIBLE_DEVICES` | GPU设备 | `0` |

### B. 相关链接

- **模型**: https://huggingface.co/guoer9/qwen3-8b-news-classifier
- **代码**: https://github.com/guoer9/money-agent/tree/vllm
- **基础模型**: https://huggingface.co/Qwen/Qwen3-8B

### C. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2024-12-24 | 初始版本，K8s部署 |

---

*文档更新: 2024-12-24*
