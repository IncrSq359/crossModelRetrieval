"""跨模态检索与评估：编码全部图文、算相似度矩阵、统计 Recall@K。"""

import torch
import time
from datetime import datetime

def encode_all(model,loader):
    """把整个数据集过一遍双塔，得到全部图文向量（推理模式）。

    Args:
        model: 训练好的 DualTowerModel。
        loader: 评估用 DataLoader（通常 shuffle=False，保持原始顺序）。
    Returns:
        img_emb: [N, D] 全部图像向量；
        txt_emb: [N, D] 全部文本向量；
        labels: list[str]，长度 N 的标签列表（与向量逐行对应）。
    """
    model.eval()
    img_list,txt_list,label_list=[],[],[]
    with torch.no_grad():                                  # 推理不需要梯度，省内存且更快
        for images,input_ids,attention_mask,labels in loader:
            img_emb,txt_emb=model(images,input_ids,attention_mask)
            img_list.append(img_emb)
            txt_list.append(txt_emb)
            label_list.extend(labels)                     # extend 摊平，避免嵌套列表
    img_emb=torch.cat(img_list,dim=0)                      # 拼接各 batch → [N,D]
    txt_emb=torch.cat(txt_list,dim=0)
    return img_emb,txt_emb,label_list

def  recall_at_k(sim_matrix,k):
    """计算 Recall@K：正确答案出现在前 K 名的查询比例。

    约定第 i 个查询的正确答案是第 i 个候选（相似度矩阵对角线）。

    Args:
        sim_matrix: [N, N] 相似度矩阵，行=查询，列=候选。
        k: 前 K 名之内命中即算成功。
    Returns:
        float：命中数 / 查询总数，取值 [0, 1]。
    """
    N=sim_matrix.size(0)
    correct=0
    for i in range(N):
        sorted_idx=torch.argsort(sim_matrix[i],descending=True)  # 相似度降序排列后的候选索引（名次表）
        if i in sorted_idx[:k].tolist():                         # 正确答案 i 是否在前 k 名
            correct+=1
    return correct/N

def evaluate(model,loader,ks=(1,5),label=""):
    """评估双向跨模态检索的 Recall@K。

    Args:
        model: 训练好的 DualTowerModel。
        loader: 评估用 DataLoader。
        ks: 要计算的 K 值列表，默认 (1, 5)。
        label: 标签，用于日志标记
    Returns:
        dict：形如 {"I2T@1": ..., "T2I@1": ..., "I2T@5": ..., "T2I@5": ...}，
        I2T 为以图搜文（查行），T2I 为以文搜图（查列，用转置矩阵）。
    """
    date = datetime.now()
    file = open(f"../data/records/Retrieve.txt","a",encoding="utf-8")
    if(label!=""):
        file.write(f"{label}\n")
    file.write(f"Time:{date}, k={ks}\n")
    t0 = time.perf_counter_ns()
    img_emb,txt_emb,labels=encode_all(model,loader)
    delta = time.perf_counter_ns()-t0
    file.write(f"Inference Cost(ns):{delta}\n")
    sim=img_emb@txt_emb.T                                 # [N,N] 跨模态余弦相似度
    result={}
    t0 = time.perf_counter_ns()
    for k in ks:
        result[f"I2T@{k}"]=recall_at_k(sim,k)
        result[f"T2I@{k}"]=recall_at_k(sim.T,k)
    delta = time.perf_counter_ns()-t0
    file.write(f"Query Cost(ns):{delta}\n")
    for key,value in result.items():
        file.write(f"{key}:{value}\n")
    file.write("\n\n")
    file.close()
    return result
