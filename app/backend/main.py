"""后端入口：FastAPI 应用，定义对外的 HTTP 接口。"""
from pathlib import Path

from app.backend import retriever
ROOT = Path(__file__).resolve().parents[2]

import torch
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

from app.backend.encoder import load_model, encode_image, encode_text
from app.backend.gallery import load_captions, build_filename_index, project_gallery
from app.backend.retriever import BruteForceRetriever

# 全局容器：启动时填充，请求时读取
state={}

@asynccontextmanager
async def lifespan(app:FastAPI):
    """启动时一次性加载模型、图库、检索器。

    Args:
        app: FastAPI 应用实例。
    """
    RESNET_PATH = ROOT / "data" / "raw" / "resnet50_timm.pth"
    PROJ_PATH = ROOT / "output" / "models" / "model.pth"
    model=load_model(RESNET_PATH,PROJ_PATH)
    model.eval()
    state["model"]=model
    
    FEATURE_PATH = ROOT / "output" / "features" / "image_features.pth"
    data = torch.load(FEATURE_PATH)
    emb=project_gallery(data["features"],model.proj_img)
    state["filenames"]=data["filenames"]
    state["emb"]=emb
    
    retriever=BruteForceRetriever(emb)
    state["retriever"]=retriever
    
    CSV_PATH=ROOT/"data"/"annotations"/"captions_flickr30k.csv"
    captions=load_captions(CSV_PATH)
    filename_index=build_filename_index(data["filenames"])
    state["captions"]=captions
    state["filename_index"]=filename_index
    yield
    state.clear()
    
    
app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
def health():
    """健康检查：返回服务当前状态，用来确认服务活着。

    Returns:
        dict：FastAPI 会自动把它转成 JSON。
    """
    n=state["emb"].shape[0]
    dim=state["emb"].shape[1]
    return {"index_size": n, "dim": dim, "retriever": "brute"}
