from torch import nn, Tensor


class feed_forward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w1 = nn.Linear(in_channels, out_channels)
        self.w2 = nn.Linear(out_channels, in_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor):
        return self.w2(self.act(self.w1(x)))
