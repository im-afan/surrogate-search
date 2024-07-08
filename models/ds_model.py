import torch
from torch import nn
from torch.distributions.normal import Normal
import snntorch as snn
from snntorch.surrogate import FastSigmoid

class DynamicSurrogateModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.theta = torch.ones((2,)) # defines mean and logstd
        self.dist = Normal(self.theta[0], self.theta[1])
        self.temp = 0
    
    def set_surrogate(self):
        self.dist = Normal(self.theta[0], torch.exp(self.theta[1]))
        self.temp = dist.sample()
        surrogate = FastSigmoid(self.temp)
        
        for module in self.model.modules():
            if(type(module) == snn.Leaky):
                module.spike_grad = surrogate

    def forward(self, x):
        self.set_surrogate()
        return self.model(x), self.dist.log_prob(self.temp)
