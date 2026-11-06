"""后端入口：FastAPI 应用，定义对外的 HTTP 接口。"""

from os import read
from pathlib import Path
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]

import torch
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

IMAGE_DIR = ROOT / "data" / "raw" / "flickr30k" / "flickr30k-images"
from PIL import Image
import io

from app.backend.encoder import load_model, encode_image, encode_text
from app.backend.gallery import load_captions, build_filename_index, project_gallery
from app.backend.retriever import BruteForceRetriever
from src.preprocess import get_image_transform
from transformers import BertTokenizer

# 全局容器：启动时填充，请求时读取
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时一次性加载模型、图库、检索器。

    Args:
        app: FastAPI 应用实例。
    """
    RESNET_PATH = ROOT / "data" / "raw" / "resnet50_timm.pth"
    PROJ_PATH = ROOT / "output" / "models" / "model.pth"
    model = load_model(RESNET_PATH, PROJ_PATH)
    model.eval()
    state["model"] = model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    state["tokenizer"] = tokenizer

    transform = get_image_transform()
    state["transform"] = transform

    FEATURE_PATH = ROOT / "output" / "features" / "image_features.pth"
    data = torch.load(FEATURE_PATH)
    emb_img = project_gallery(data["features"], model.proj_img)
    state["filenames"] = data["filenames"]
    state["emb_img"] = emb_img

    retriever = BruteForceRetriever(emb_img)
    state["retriever"] = retriever

    CSV_PATH = ROOT / "data" / "annotations" / "captions_flickr30k.csv"
    captions = load_captions(CSV_PATH)
    filename_index = build_filename_index(data["filenames"])
    state["captions"] = captions
    state["filename_index"] = filename_index

    # 预热
    text = "a dog"
    emb_txt = encode_text(model, state["tokenizer"], text)
    print("文本编码器预热完成，输出形状：", emb_txt.shape)
    yield
    state.clear()


class TextQuery(BaseModel):
    """文本搜索的请求体。

    Attributes:
        query: 查询文本。
        top_k: 返回条数，默认 20。
    """

    query: str
    top_k: int = 20


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
def health():
    """健康检查：返回服务当前状态，用来确认服务活着。

    Returns:
        dict：FastAPI 会自动把它转成 JSON。
    """
    n = state["emb_img"].shape[0]
    dim = state["emb_img"].shape[1]
    return {"index_size": n, "dim": dim, "retriever": "brute"}


def build_response(q_vec, top_k, t0):
    """由查询向量得到完整响应（文本搜图、图搜图共用）。

    Args:
        q_vec: 已投影并 L2 归一化的查询向量，形状 [256]。
        top_k: 返回条数。
        t0: 请求开始时间，由 time.perf_counter() 得到。

    Returns:
        符合接口契约的字典。
    """
    idx, scores = state["retriever"].search_vector(q_vec, top_k)
    idx = idx.tolist()
    scores = scores.tolist()
    results = []
    for i, s in zip(idx, scores):
        fn = state["filenames"][i]
        results.append(
            {
                "filename": fn,
                "score": round(s, 4),
                "url": f"/images/{fn}",
                "captions": state["captions"][fn],
            }
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"latency_ms": latency_ms, "retriever": "brute", "results": results}


@app.post("/api/search/text")
def search_txt(body: TextQuery):
    """以文搜图：输入一句文本，返回最相似的图片。

    Args:
        body: 请求体，含 query（查询文本）和 top_k（返回条数）。

    Returns:
        build_response 组装好的字典。
    """
    t0 = time.perf_counter()
    q_vec = encode_text(state["model"], state["tokenizer"], body.query)
    return build_response(q_vec, body.top_k, t0)


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 20):
    """以图搜图：上传一张图片，返回最相似的图片。

    Args:
        file: 上传的图片文件（jpg 或 png）。
        top_k: 返回条数，默认 20。

    Returns:
        build_response 组装好的字典。
    """
    t0 = time.perf_counter()
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    q_vec = encode_image(state["model"], state["transform"], img)
    return build_response(q_vec, top_k, t0)


app.mount("/images", StaticFiles(directory=IMAGE_DIR))
