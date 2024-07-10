import torch

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
        grad = 1/2 * ctx.width * torch.pow(2/(torch.exp(ctx.width*mem) + torch.exp(-ctx.width*mem)), 2)
        return grad, None
    
def tanh_surrogate(width=25):
    width = width
    def inner(x):
        return ATanSurrogate.apply(x, width)
    return inner

