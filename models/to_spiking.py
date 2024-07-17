import torch
from torch import nn
from snntorch import Leaky, BatchNormTT2d
from snntorch import utils
from .batchnorm import tdBatchNorm2d

class SNN(nn.Module):
    def __init__(self, module: nn.Module, beta=0.5, num_steps=5):
        super().__init__()
        self.num_steps = num_steps
        self.module = module
        self.to_spiking(module)

    def to_spiking(self, module: nn.Module, beta=0.5, num_steps=5):
        #print("recurse")
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                #print("convert relu")
                setattr(module, name, Leaky(beta=0.5, init_hidden=True)) 
            elif isinstance(child_module, nn.BatchNorm2d):
                #setattr(module, name, tdBatchNorm2d(bn=child_module, alpha=1))
                #setattr(module, name, BatchNormTT2d(input_features=child_module.num_features, time_steps=num_steps))
                pass

            self.to_spiking(child_module, beta=beta, num_steps=num_steps)

    def reset(self, module):
        #print("reset ", module)
        utils.reset(module)
        for name, child_module in module.named_children():
            if(isinstance(child_module, Leaky)):
                continue
            #print("named child ", child_module)
            self.reset(child_module)

    def forward(self, input):
        spk_rec = []
        self.reset(self.module)
        for step in range(self.num_steps):
            #spk_out, mem_out = net(data[step])
            spk_out = self.module(input[step])
            spk_rec.append(spk_out)
            #mem_rec.append(mem_out)

        return torch.stack(spk_rec)#, torch.stack(mem_rec)
