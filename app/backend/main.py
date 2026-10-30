"""后端入口：FastAPI 应用，定义对外的 HTTP 接口。"""
from fastapi import FastAPI

app=FastAPI()
@app.get("/api/health")

def health():
    """健康检查：返回服务当前状态，用来确认服务活着。

    Returns:
        dict：FastAPI 会自动把它转成 JSON。
    """
    return {"index_size":0,"dim":256,"retriever":"brute"}