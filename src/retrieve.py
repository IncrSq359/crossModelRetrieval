import torch

def encode_all(model,loader):
    model.eval()
    img_list,txt_list,label_list=[],[],[]
    with torch.no_grad():
        for images,input_ids,attention_mask,labels in loader:
            img_emb,txt_emb=model(images,input_ids,attention_mask)
            img_list.append(img_emb)
            txt_list.append(txt_emb)
            label_list.extend(labels)
    img_emb=torch.cat(img_list,dim=0)
    txt_emb=torch.cat(txt_list,dim=0)
    return img_emb,txt_emb,label_list

def  recall_at_k(sim_matrix,k):
    N=sim_matrix.size(0)
    correct=0
    for i in range(N):
        sorted_idx=torch.argsort(sim_matrix[i],descending=True)
        if i in sorted_idx[:k].tolist():
            correct+=1
    return correct/N

def evaluate(model,loader,ks=(1,5)):
    img_emb,txt_emb,labels=encode_all(model,loader)
    sim=img_emb@txt_emb.T
    result={}
    for k in ks:
        result[f"I2T@{k}"]=recall_at_k(sim,k)
        result[f"T2I@{k}"]=recall_at_k(sim.T,k)
    return result