import torch
import numpy as np

class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          ######### Your code here

          ctx.save_for_backward(mem)
          spk = torch.heaviside(mem, torch.zeros(1))
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (spk,) = ctx.saved_tensors  # retrieve the membrane potential
          ######### Your code here
          #This represents legal code (aka membrane_potential is accessable, but you need to modify it)
          #grad = mem
          grad = 1/np.pi * 1/(1 + (spk*np.pi)**2)
          return grad


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

class TanhSurrogate(torch.autograd.Function):
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
        grad = 1/2 * ctx.width * torch.pow(2/(torch.exp(ctx.width*mem) + torch.exp(-ctx.width*mem)), 2) * grad_input
        return grad, None
    
def tanh_surrogate(width=0.5):
    width = width
    def inner(x):
        return ATanSurrogate.apply(x, width)
    return inner

def tanh_surrogate1(width=0.5):
    def tanh_grad(input_, grad_input, spikes):
        temp = width
        grad = 1/2 * temp * torch.pow(2 / (torch.exp(temp*input_) + torch.exp(-temp*input_)), 2) * grad_input
        return grad
    
    return tanh_grad