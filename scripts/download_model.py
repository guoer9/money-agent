"""预下载模型脚本 - 用于Docker构建或离线环境"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MODEL_NAME, MODEL_PATH


def download_model():
    """下载并缓存模型"""
    print(f"正在下载模型: {MODEL_NAME}")
    
    # 下载tokenizer
    print("下载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 下载模型
    print("下载模型权重...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # 可选: 保存到本地目录
    if MODEL_PATH:
        os.makedirs(MODEL_PATH, exist_ok=True)
        print(f"保存模型到: {MODEL_PATH}")
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    
    print("模型下载完成!")
    print(f"模型标签: {model.config.id2label}")


if __name__ == "__main__":
    download_model()
