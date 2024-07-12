import torch 
import torch.nn as nn 
import snntorch as snn
from torch.autograd import Function



class ATanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, width):
        ctx.save_for_backward(mem)
        ctx.width = width
        spk = (mem > 0).float()
        return spk
    
    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (ctx.width/torch.pi) / (1 + (ctx.width*mem)) * grad_input
        return grad, None
    
def atan_surrogate(width=25):
    width = width
    def inner(x):
        return ATanSurrogate.apply(x, width)
    return inner

def atan_surrogate1(width=25):
    def atan_grad(input_, grad_input, spikes):
        mem = input_
        grad = (width/torch.pi) / (1 + (width*mem)) * grad_input
        return grad
    return atan_grad


class DSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, b, c):
        ctx.save_for_backward(mem)
        ctx.b = b
        ctx.c = c
        
        spk = (mem > 0).float()
        return spk

    @staticmethod
    def backward(ctx, grad_output):
        mem, = ctx.saved_tensors
        b = ctx.b
        c = ctx.c

        a = 1 / (torch.tanh(b * (1 - c)) - torch.tanh(b * (-c)))
        grad_input = a * b * (1 - torch.tanh(b * (mem - c))**2) * grad_output
        
        return grad_input, None, None

def dspike(b=1.0):
    c = 0.5
    def inner(mem):
        return DSpikeFunction.apply(mem, b, c)
    return inner

def dspike1(b=1.0):
    b = torch.tensor(b)
    c = torch.tensor(0.5)
    def dspike_grad(input_, grad_input, spikes):
        a = 1 / (torch.tanh(b * (1 - c)) - torch.tanh(b * (-c)))
        grad = a * b * (1 - torch.tanh(b * (input_ - torch.ones_like(input_)*c))**2) * grad_input
        
        return grad
    return dspike_grad
