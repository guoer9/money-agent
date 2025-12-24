"""FastAPI 主应用 - 中文金融新闻情绪分析"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .model import get_model, FinBERTSentiment
from .schemas import (
    TextInput, 
    BatchTextInput, 
    SentimentResult, 
    BatchSentimentResult,
    HealthCheck
)
from .config import API_HOST, API_PORT, DEVICE


# 全局模型实例
model: FinBERTSentiment = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global model
    print("正在加载中文金融情绪分析模型...")
    model = get_model()
    print("模型加载完成!")
    yield
    print("服务关闭...")


app = FastAPI(
    title="FinBERT 金融新闻情绪分析 API",
    description="基于FinBERT的中文金融新闻情绪识别服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """API根路径"""
    return {
        "message": "FinBERT 金融新闻情绪分析 API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthCheck(
        status="ok",
        model_loaded=model is not None,
        device=str(model.device) if model else DEVICE
    )


@app.post("/predict", response_model=SentimentResult, tags=["Prediction"])
async def predict_sentiment(input_data: TextInput):
    """单条文本情绪分析"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = model.predict(input_data.text)
        return SentimentResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchSentimentResult, tags=["Prediction"])
async def predict_batch(input_data: BatchTextInput):
    """批量文本情绪分析"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = model.predict_batch(input_data.texts)
        return BatchSentimentResult(
            results=[SentimentResult(**r) for r in results],
            total=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
