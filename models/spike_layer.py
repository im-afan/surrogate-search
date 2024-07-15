import torch
from torch import nn

class SpikeLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = []
        for i in range(x.shape[0]):
            out.append(self.module(x[i]))
        out = torch.stack(out)
        return out