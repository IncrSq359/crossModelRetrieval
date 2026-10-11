"""对比学习训练循环：只更新双塔的投影层。"""

import torch
from loss import dual_infonce_loss

def train(model,loader,num_epoches=30,lr=1e-3):
    """用 InfoNCE 训练双塔模型的投影层。

    每个 batch 固定走"清空梯度 → 前向 → 算损失 → 反向 → 更新参数"
    五步。编码器已冻结，优化器只管理 requires_grad=True 的参数。

    Args:
        model: DualTowerModel 实例。
        loader: 训练 DataLoader，建议用 class-balanced 采样器组 batch。
        num_epoches: 遍历整个数据集的轮数，默认30。
        lr: 学习率（Adam），默认1e-3。
    Returns:
        list[float]：每个 epoch 的平均损失，可用于绘制 loss 曲线。
    """
    trainable_params=[p for p in model.parameters() if  p.requires_grad]  # 只保留投影层参数
    optimizer=torch.optim.Adam(trainable_params,lr=lr)
    model.train()
    loss_history=[]
    for epoch in range(num_epoches):
        epoch_loss=0.0
        for images,input_ids,attention_mask,_ in loader:
            optimizer.zero_grad()                              # 清空上一batch的梯度（梯度默认累加）
            img_emb,txt_emb=model(images,input_ids,attention_mask)  # 前向（训练需保留计算图，不能no_grad）
            loss=dual_infonce_loss(img_emb,txt_emb)
            loss.backward()                                    # 反向传播：计算各可训练参数的梯度
            optimizer.step()                                   # 优化器按梯度更新投影层
            epoch_loss+=loss.item()                            # .item() 取纯数值并释放计算图
        avg_loss=epoch_loss/len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{num_epoches}, Loss: {avg_loss:.4f}")
    return loss_history
