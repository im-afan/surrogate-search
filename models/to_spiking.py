import torch
from torch import nn
from snntorch import Leaky, BatchNormTT2d
from .batchnorm import tdBatchNorm2d

def to_spiking(module: nn.Module, beta=0.5, num_steps=5):
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Sequential):
            to_spiking(child_module)
        elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
            setattr(module, name, Leaky(beta=0.5, init_hidden=True)) 
        elif isinstance(child_module, nn.BatchNorm2d):
            #setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))
            #setattr(module, name, BatchNormTT2d(input_features=child_module.num_features, time_steps=num_steps))
            pass
