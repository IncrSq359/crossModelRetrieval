import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class layer_norm(nn.Module):
    """Layer Normalization,对H*W的每个元素单独处理\n
    config = 1 --> [N,C,H,W]\n
    config = 2 --> [N,H,W,C] 或 [N,C]

    Args:
        dim: normalized_shape
        config: tensor_shape
        eps
    """

    def __init__(self, dim, config, eps=1e-6):
        super().__init__()
        if config not in (1, 2):
            raise NotImplementedError
        self.config = config
        self.norm_shape = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor):
        if self.config == 1:
            ave = x.mean(dim=1, keepdim=True)
            s = (x - ave).pow(2).mean(dim=1, keepdim=True)
            x = (x - ave) / (s + self.eps).sqrt()
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, (self.norm_shape,), self.weight, self.bias, self.eps)


class Block(nn.Module):
    """ConvNeXt Block \n
    点卷积在维度变换后使用线性层实现，起到加速作用\n
    GELU激活函数默认使用tanh近似来加速运算，在低精度(FP16)环境下可能出现tanh饱和或溢出，可取消近似(none)

    Args:
        gelu_config: 'tanh'|'none'
    """

    def __init__(self, dim, layer_scale_value=1e-6, gelu_config='tanh'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = layer_norm(dim, 2)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        if gelu_config not in ['tanh', 'none']:
            raise NotImplementedError
        self.act = nn.GELU(approximate=gelu_config)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x: Tensor):
        y = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N,C,H,W] -> [N,H,W,C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N,H,W,C] -> [N,C,H,W]
        x = x + y
        return x


class ConvNeXt(nn.Module):
    """采用ConvNeXt_T模型，预训练权重下载：https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth \n
    理论精度较ResNet-50高出3%,较Swin-Transformer高出1%\n
    在未进行fine-tune时应将out_class设为0,此时仅输出特征向量,可同模态内做相似度/检索/聚类\n
    参数命名与官方权重保持一致(downsample_layers/stages/norm/head)，可直接load_state_dict
    (out_class=0时需 strict=False，以忽略官方权重中的head.*)
    Args:
        in_channel: 输入通道，通常为RGB(3)或RGBW(4)
        out_class: 输出类别,在为0时分类层为空仅输出特征，适合在未fine-tune时作相似度对比\n
        dim: 不同阶段的通道数
        depth: 每个阶段内的Block堆叠数
        layer_scale_value: Layer Scale初始值
        head_init_scale: 分类头初始权重缩放系数
    """

    def __init__(self, in_channel=3, out_class=1000, dim=[96, 192, 384, 768],
                 depth=[3, 3, 9, 3], layer_scale_value=1e-6, head_init_scale=0.001):
        super().__init__()
        # downsample_layers[0] 为 stem，[1..3] 为阶段之间的下采样
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channel, dim[0], kernel_size=4, stride=4),
            layer_norm(dim[0], config=1)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                layer_norm(dim[i], config=1),
                nn.Conv2d(dim[i], dim[i + 1], kernel_size=2, stride=2)
            ))
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim[i], layer_scale_value=layer_scale_value, gelu_config='tanh')
                for _ in range(depth[i])
            ])
            self.stages.append(stage)
        self.norm = layer_norm(dim[3], config=2)
        if out_class > 0:
            self.head = nn.Linear(dim[3], out_class)
        else:
            self.head = nn.Identity()
        self.apply(self.init_weight)
        if out_class > 0:
            with torch.no_grad():
                self.head.weight.mul_(head_init_scale)
                self.head.bias.mul_(head_init_scale)

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([2, 3]))  # 全局平均池化 [N,C,H,W] -> [N,C]
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = ConvNeXt(out_class=0).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)  # torch.Size([2, 768])
    model = ConvNeXt(out_class=1000).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    print(out.shape)  # torch.Size([2, 1000])