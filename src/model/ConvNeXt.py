import torch
from torch import nn
from torch import Tensor


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
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.pwconv2 = nn.Linear(4*dim, dim)
        if (gelu_config not in ['tanh', 'none']):
            raise NotImplementedError
        self.act = nn.GELU(gelu_config)
        self.norm = layer_norm(dim, 2)
        if (layer_scale_value > 0):
            self.gamma = nn.Parameter(
                layer_scale_value*torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x: Tensor):
        y = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act()
        x = self.pwconv2(x)
        if (self.gamma is not None):
            x = self.gamma*x
        x = x.permute(0, 3, 1, 2)
        x = x+y
        return x


class ConvNeXt(nn.Module):
    """采用ConvNeXt_T模型，预训练权重下载：https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth \n
    理论精度较ResNet-50高出3%,较Swin-Transformer高出1%\n
    在未进行fine-tune时应将out_calss设为0,此时仅输出权重,可同模态内做相似度/检索/聚类
    Args:
        in_channel: 输入通道，通常为RGB(3)或RGBW(4)
        out_calss: 输出类别,在为0时分类层为空仅输出权重，适合在未fine-tune时作相似度对比\n
        dim: 不同块的通道数
        depth: 一阶段内的块的堆叠数
        layer_scale_value: 权重缩放值
        head_init_scale: 分类头初始权重缩放系数
    """

    def __init__(self, in_channel=3, out_calss=1000, dim=[96, 192, 384, 768],
                 depth=[3, 3, 9, 3], layer_scale_value=1e-6, head_init_scale=0.001):
        super().__init__()
        self.down_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channel, dim[0], kernel_size=4, stride=4),
            layer_norm(dim[0], config=1)
        )
        self.down_layers.append(stem)
        for i in range(3):
            self.down_layers.append(nn.Sequential(
                layer_norm(dim[i], config=1),
                nn.Conv2d(dim[i], dim[i+1], kernel_size=2, stride=2)
            ))
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential()
            for j in depth[i]:
                stage.append(
                    Block(dim[i], layer_scale_value=layer_scale_value, gelu_config='tanh'))
            self.stages.append(stage)
        self.norm = layer_norm(dim[3], config=2)
        if (out_calss > 0):
            self.head = nn.Linear(dim[3], out_calss)
        else:
            self.head = nn.Identity()
        with torch.no_grad():
            self.apply(self.init_weight)
            self.head.weight *= head_init_scale
            self.head.bias *= head_init_scale

    def init_weight(self, x):
        if isinstance(x, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(x.weight, 0, 0.02, a=-0.04, b=0.04)
            nn.init.constant_(x.bias, 0)

    def forward(self, x: Tensor):
        for i in range(4):
            x = self.down_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([2, 3]))
        x = self.head(x)
        return x


class layer_norm(nn.Module):
    """Layer Normalization,对H*W的每个元素单独处理\n
    config = 1 --> [N,C,H,W]\n
    config = 2 --> [N,H,W,C]

    Args:
        dim: normalized_shape
        config: tensor_shape
        eps
    """

    def __init__(self, dim, config, eps=1e-6):
        super().__init__()
        self.config = config
        self.norm_shape = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor):
        if (self.config == 1):
            ave = x.mean(dim=1, keepdim=True)
            s = (x-ave).pow(2).mean(dim=1, keepdim=True)
            x = (x-ave)/(s+self.eps).sqrt()
            x = x*self.weight[:, None, None]+self.bias[:, None, None]
            return x
        elif (self.config == 2):
            return nn.LayerNorm(self.norm_shape, self.eps, elementwise_affine=True)
