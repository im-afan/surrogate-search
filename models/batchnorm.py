import torch
from torch import nn

class tdBatchNorm2d(nn.BatchNorm2d):
    """Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, bn: nn.BatchNorm2d, alpha: float):
        super(tdBatchNorm2d, self).__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.alpha = alpha
        self.V_th = 0.5
        # self.weight.data = bn.weight.data
        # self.bias.data = bn.bias.data
        # self.running_mean.data = bn.running_mean.data
        # self.running_var.data = bn.running_var.data


    def forward(self, input):
        # Ensure correct behavior in both training and evaluation modes
        if self.training or not self.track_running_stats:
            mean = input.mean(dim=[0, 2, 3], keepdim=True)
            var = input.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
            
            if self.training:
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)

        # Normalize input
        input = self.alpha * self.V_th * (input - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation if affine=True
        if self.affine:
            input = input * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        return input

    def forward1(self, input):
        if False:
            # compulsory eval mode for normal bn
            self.training = False
            return super().forward(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            print(input.shape)
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        channel_dim = input.shape[2]
        input = self.alpha * self.V_th * (input - mean.reshape(1, 1, channel_dim, 1, 1)) / \
                (torch.sqrt(var.reshape(1, 1, channel_dim, 1, 1) + self.eps))
        if self.affine:
            input = input * self.weight.reshape(1, 1, channel_dim, 1, 1) + self.bias.reshape(1, 1, channel_dim, 1, 1)

        return input
