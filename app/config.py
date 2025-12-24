"""配置文件"""
import os
import torch

# 设置HuggingFace镜像源 (国内加速)
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# 模型配置 - 中文金融情绪分析预训练模型
MODEL_NAME = os.getenv("MODEL_NAME", "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/finbert-chinese")

# 情绪标签映射
SENTIMENT_LABELS_ZH = {
    "negative": "负面",
    "neutral": "中性",
    "positive": "正面",
    "Negative": "负面",
    "Neutral": "中性",
    "Positive": "正面",
    "LABEL_0": "负面",
    "LABEL_1": "中性",
    "LABEL_2": "正面",
    0: "负面",
    1: "中性",
    2: "正面"
}

# API配置
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8888"))

# 设备配置
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
