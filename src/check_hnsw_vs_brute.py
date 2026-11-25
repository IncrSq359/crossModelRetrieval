"""对比 BruteForceRetriever 与 HNSWRetriever 的 Top-10 重叠率。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from app.backend import paths
from app.backend.encoder import load_model
from app.backend.gallery import project_gallery
from app.backend.retriever import BruteForceRetriever, HNSWRetriever

CONVNEXT_PATH = ROOT / "data" / "raw" / "convnext_tiny_1k_224_ema.pth"
PROJ_PATH = ROOT / "output" / "models" / "projection_flickr30k.pth"
model = load_model(CONVNEXT_PATH, PROJ_PATH)
model.eval()
FEATURE_PATH = ROOT / "output" / "features" / "image_features.pth"
data = torch.load(FEATURE_PATH)
emb_img = project_gallery(data["features"], model.proj_img)

brute = BruteForceRetriever(emb_img)
hnsw = HNSWRetriever(emb_img)

n = emb_img.shape[0]
rows = torch.randperm(n)[:500]
queries = emb_img[rows]

total_overlap = 0
for q in queries:
    idx_b, _ = brute.search_vector(q, 10)
    idx_h, _ = hnsw.search_vector(q, 10)
    set_b = set(idx_b.tolist())
    set_h = set(idx_h.tolist())
    total_overlap += len(set_b & set_h)
print(total_overlap / (500 * 10))
