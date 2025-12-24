"""模型加载和推理 - 金融情绪分析"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import os

from .config import MODEL_NAME, SENTIMENT_LABELS_ZH, DEVICE


class FinBERTSentiment:
    """FinBERT 金融情绪分析模型"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        self.device = torch.device(DEVICE)
        self.model = None
        self.tokenizer = None
        
    def load(self, model_path: str = None):
        """加载预训练的FinBERT模型"""
        load_path = model_path or self.model_name
        
        print(f"Loading FinBERT from: {load_path}")
        print(f"Using device: {self.device}")
        
        # 支持离线模式和在线模式
        local_only = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=local_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            local_files_only=local_only,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        print(f"Model labels: {self.id2label}")
        print("Model loaded successfully!")
        
    def predict(self, text: str) -> Dict:
        """单条文本情绪预测"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        probs_np = probs.cpu().numpy()[0]
        pred_idx = probs_np.argmax()
        pred_label = self.id2label.get(pred_idx, str(pred_idx))
        
        return {
            "text": text,
            "sentiment": pred_label,
            "sentiment_zh": SENTIMENT_LABELS_ZH.get(pred_label, pred_label),
            "confidence": float(probs_np[pred_idx]),
            "probabilities": {
                self.id2label.get(i, str(i)): float(p) 
                for i, p in enumerate(probs_np)
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """批量文本情绪预测"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
            probs_np = probs.cpu().numpy()
            
            for j, text in enumerate(batch_texts):
                pred_idx = probs_np[j].argmax()
                pred_label = self.id2label.get(pred_idx, str(pred_idx))
                results.append({
                    "text": text,
                    "sentiment": pred_label,
                    "sentiment_zh": SENTIMENT_LABELS_ZH.get(pred_label, pred_label),
                    "confidence": float(probs_np[j][pred_idx]),
                    "probabilities": {
                        self.id2label.get(k, str(k)): float(p)
                        for k, p in enumerate(probs_np[j])
                    }
                })
                
        return results


_model = None


def get_model() -> FinBERTSentiment:
    """获取模型单例"""
    global _model
    if _model is None:
        _model = FinBERTSentiment()
        _model.load()
    return _model
