"""相似度模块：计算余弦相似度矩阵，并按"同类/不同类"统计平均相似度。"""

import torch

def cosine_two_vectors(a,b):
    """计算两个一维向量之间的余弦相似度。

    Args:
        a: 一维张量（向量）。
        b: 一维张量（向量），长度需与 a 相同。

    Returns:
        标量余弦相似度 = 点积 / (两向量模长之积)，取值范围 [-1, 1]。
    """
    dot=torch.sum(a*b)
    norm_a=torch.sqrt(torch.sum(a**2))
    norm_b=torch.sqrt(torch.sum(b**2))
    return dot/(norm_a*norm_b)

def cosine_similarity_matrix(features):
    """一次性计算所有样本两两之间的余弦相似度矩阵。

    先把每个特征向量按行归一化（除以自身模长），再用矩阵乘法
    一次得到全部两两余弦值，比双重循环快得多。

    Args:
        features: 特征张量，形状 [样本数, 特征维度]。

    Returns:
        相似度矩阵，形状 [样本数, 样本数]，元素 [i,j] 即样本 i 与 j 的余弦相似度；
        对角线为 1（自己比自己），矩阵对称。
    """
    norms=torch.sqrt(torch.sum(features**2,dim=1,keepdim=True))
    normed=features/norms
    sim=normed@normed.T
    return sim

def analyze_similarity(sim,labels):
    """统计同类样本对与不同类样本对的平均相似度。

    遍历矩阵上三角（每对只算一次），按标签是否相同分成两组求平均。

    Args:
        sim: cosine_similarity_matrix 返回的相似度矩阵。
        labels: 与矩阵顺序对应的标签列表。

    Returns:
        四元组：
        same_mean: 同类样本对的平均相似度；
        diff_mean: 不同类样本对的平均相似度；
        n_same: 同类样本对数量；
        n_diff: 不同类样本对数量。
    """
    same_values=[]
    diff_values=[]
    n=len(labels)
    for i in range(n):
        for j in range(i+1,n):
            value=sim[i,j].item()
            if labels[i]==labels[j]:
                same_values.append(value)
            else:
                diff_values.append(value)
    
    same_mean=sum(same_values)/len(same_values)
    diff_mean=sum(diff_values)/len(diff_values)
    return same_mean,diff_mean,len(same_values),len(diff_values)
