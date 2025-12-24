"""Pydantic数据模型"""
from pydantic import BaseModel, Field
from typing import List


class TextInput(BaseModel):
    """单条文本输入"""
    text: str = Field(..., description="待分析的金融新闻文本", min_length=1, max_length=512)


class BatchTextInput(BaseModel):
    """批量文本输入"""
    texts: List[str] = Field(..., description="待分析的文本列表", min_length=1)


class SentimentResult(BaseModel):
    """情绪分析结果"""
    text: str = Field(..., description="原始文本")
    sentiment: str = Field(..., description="情绪标签(英文)")
    sentiment_zh: str = Field(..., description="情绪标签(中文)")
    confidence: float = Field(..., description="置信度", ge=0, le=1)
    probabilities: dict = Field(..., description="各类别概率分布")


class BatchSentimentResult(BaseModel):
    """批量情绪分析结果"""
    results: List[SentimentResult]
    total: int


class HealthCheck(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    model_loaded: bool
    device: str
