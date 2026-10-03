import torch

def cosine_two_vectors(a,b):
    dot=torch.sum(a*b)
    norm_a=torch.sqrt(torch.sum(a**2))
    norm_b=torch.sqrt(torch.sum(b**2))
    return dot/(norm_a*norm_b)

def cosine_similarity_matrix(features):
    norms=torch.sqrt(torch.sum(features**2,dim=1,keepdim=True))
    normed=features/norms
    sim=normed@normed.T
    return sim

def analyze_similarity(sim,labels):
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


if __name__ == "__main__":
    from image_feature import load_features           # 导入之前写的读取函数

    # 读入100张图的真实特征（不用再跑ResNet）
    features, labels, paths = load_features("output/features/image_features.pth")
    print("特征形状：", features.shape)                 # [100, 2048]

    sim = cosine_similarity_matrix(features)
    print("相似度矩阵形状：", sim.shape)                # [100, 100]
    print("对角线（自己比自己）：", sim.diag()[:5])     # 应全是1
    print("是否对称：", torch.allclose(sim, sim.T))    # True，因为cos(a,b)=cos(b,a)
    print("数值范围：", sim.min().item(), sim.max().item())  # 应落在[-1,1]
    same_mean, diff_mean, n_same, n_diff = analyze_similarity(sim, labels)
    print("同类对数：", n_same, " 平均相似度：", same_mean)
    print("不同类对数：", n_diff, " 平均相似度：", diff_mean)