import torch
from torch import nn
from snntorch import Leaky, BatchNormTT2d
from snntorch import utils
from .spike_layer import SpikeLayer, tdBatchNorm2d

class SNN(nn.Module):
    def __init__(self, module: nn.Module, beta=0.5, num_steps=5):
        super().__init__()
        self.module = module
        self.to_spiking(self.module, beta=beta, num_steps=num_steps)

    def to_spiking(self, module: nn.Module, beta=0.5, num_steps=5):
        #print("recurse")
        children = 0
        for name, child_module in module.named_children():
            children += 1

            if isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                setattr(module, name, Leaky(beta=0.5, init_hidden=True)) 
            elif isinstance(child_module, nn.BatchNorm2d):
                setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))

            child_children = self.to_spiking(child_module, beta=beta, num_steps=num_steps)

            if(child_children == 0 and not isinstance(getattr(module, name), tdBatchNorm2d)):
                print(f"convert {child_module} to spiking")
                setattr(module, name, SpikeLayer(getattr(module, name)))


        return children

    def forward(self, x):
        utils.reset(self.module)  # resets hidden states for all LIF neurons in net
        return self.module(x)