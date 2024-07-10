import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F

class SpikingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.5

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.flatten = nn.Flatten()

    def forward(self, x, batch_size=32):

        # Initialize hidden states and outputs at t=0
        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1 = self.lif1(cur1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2 = self.lif2(cur2)

        #print(spk2.shape, self.flatten(spk2).shape)
        cur3 = self.fc1(self.flatten(spk2))
        spk3, mem3 = self.lif3(cur3)

        return spk3, mem3