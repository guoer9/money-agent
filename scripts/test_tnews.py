"""使用TNEWS公共数据集测试情绪分析模型"""
import requests
import json
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

API_URL = "http://localhost:8888"


def load_tnews_data(sample_size=100):
    """加载TNEWS数据集金融相关子集"""
    print("正在从HuggingFace加载TNEWS数据集...")
    
    dataset = load_dataset("clue", "tnews")
    
    # TNEWS标签索引: 4-财经(104), 7-股票(108) 为金融相关
    label_names = dataset["train"].features["label"].names
    print(f"标签列表: {label_names}")
    
    finance_indices = []
    for i, name in enumerate(label_names):
        if name in ["104", "108"]:  # 104-财经, 108-股票
            finance_indices.append(i)
    print(f"金融类别索引: {finance_indices}")
    
    test_data = dataset["validation"]
    finance_news = [
        item["sentence"] for item in test_data 
        if item["label"] in finance_indices
    ]
    
    print(f"金融新闻总数: {len(finance_news)}")
    
    if sample_size and len(finance_news) > sample_size:
        finance_news = finance_news[:sample_size]
    
    return finance_news


def test_sentiment_api(texts):
    """测试情绪分析API"""
    results = []
    
    print(f"\n正在测试 {len(texts)} 条金融新闻...")
    
    for text in tqdm(texts):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=10
            )
            if resp.status_code == 200:
                results.append(resp.json())
            else:
                print(f"API错误: {resp.status_code}")
        except Exception as e:
            print(f"请求失败: {e}")
    
    return results


def analyze_results(results):
    """分析测试结果"""
    print("\n" + "="*60)
    print("TNEWS金融新闻情绪分析测试结果")
    print("="*60)
    
    sentiments = [r["sentiment"] for r in results]
    sentiment_counts = Counter(sentiments)
    
    print(f"\n总测试样本: {len(results)}")
    print("\n情绪分布:")
    for sentiment, count in sentiment_counts.most_common():
        zh = {"Positive": "正面", "Negative": "负面", "Neutral": "中性"}.get(sentiment, sentiment)
        pct = count / len(results) * 100
        print(f"  {zh} ({sentiment}): {count} ({pct:.1f}%)")
    
    confidences = [r["confidence"] for r in results]
    avg_conf = sum(confidences) / len(confidences)
    print(f"\n平均置信度: {avg_conf:.4f}")
    
    print("\n" + "="*60)
    print("示例预测结果 (前10条)")
    print("="*60)
    
    for r in results[:10]:
        zh = {"Positive": "正面", "Negative": "负面", "Neutral": "中性"}.get(r["sentiment"], r["sentiment"])
        print(f"\n文本: {r['text'][:60]}...")
        print(f"情绪: {zh} | 置信度: {r['confidence']:.4f}")


def main():
    print("="*60)
    print("FinBERT中文金融情绪分析 - TNEWS数据集测试")
    print("="*60)
    
    try:
        resp = requests.get(f"{API_URL}/health")
        health = resp.json()
        print(f"\nAPI状态: {health['status']}")
        print(f"模型已加载: {health['model_loaded']}")
        print(f"运行设备: {health['device']}")
    except Exception as e:
        print(f"无法连接API: {e}")
        print("请确保服务已启动: uvicorn app.main:app --port 8888")
        return
    
    texts = load_tnews_data(sample_size=50)
    results = test_sentiment_api(texts)
    
    if results:
        analyze_results(results)
    else:
        print("没有获取到测试结果")


if __name__ == "__main__":
    main()
