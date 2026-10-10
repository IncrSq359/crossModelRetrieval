import torch
from loss import dual_infonce_loss

def train(model,loader,num_epoches=30,lr=1e-3):
    trainable_params=[p for p in model.parameters() if  p.requires_grad]
    optimizer=torch.optim.Adam(trainable_params,lr=lr)
    model.train()
    loss_history=[]
    for epoch in range(num_epoches):
        epoch_loss=0.0
        for images,input_ids,attention_mask,_ in loader:
            optimizer.zero_grad()
            img_emb,txt_emb=model(images,input_ids,attention_mask)
            loss=dual_infonce_loss(img_emb,txt_emb)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        avg_loss=epoch_loss/len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{num_epoches}, Loss: {avg_loss:.4f}")
    return loss_history