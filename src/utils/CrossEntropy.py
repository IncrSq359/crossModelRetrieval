import torch


def cross_entropy(pred, target):
    # 数值稳定性处理
    pred = pred - pred.max(dim=1, keepdim=True).values

    # softmax
    exp_pred = torch.exp(pred)
    prob = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)

    correct_prob = prob[torch.arange(pred.size(0)), target]

    loss = -torch.log(correct_prob)

    return loss.mean()
