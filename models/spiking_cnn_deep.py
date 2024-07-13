import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import snntorch as snn
from snntorch import surrogate

class SpikingCNNDeep(nn.Module):
    def __init__(self):
        super().__init__()

        # Define layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = snn.Leaky(beta=0.5, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = snn.Leaky(beta=0.5, init_hidden=True)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = snn.Leaky(beta=0.5, init_hidden=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = snn.Leaky(beta=0.5, init_hidden=True)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = snn.Leaky(beta=0.5, init_hidden=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = snn.Leaky(beta=0.5, init_hidden=True)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.relu7 = snn.Leaky(beta=0.5, init_hidden=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu8 = snn.Leaky(beta=0.5, init_hidden=True)
        self.fc3 = nn.Linear(512, 10)
        self.relu9 = snn.Leaky(beta=0.5)

    def forward(self, input_spikes):
        spike_out = self.conv1(input_spikes)
        spike_out = self.relu1(spike_out)
        spike_out = self.conv2(spike_out)
        spike_out = self.relu2(spike_out)
        spike_out = self.pool1(spike_out)
        spike_out = self.conv3(spike_out)
        spike_out = self.relu3(spike_out)
        spike_out = self.conv4(spike_out)
        spike_out = self.relu4(spike_out)
        spike_out = self.pool2(spike_out)
        spike_out = self.conv5(spike_out)
        spike_out = self.relu5(spike_out)
        spike_out = self.conv6(spike_out)
        spike_out = self.relu6(spike_out)
        spike_out = self.pool3(spike_out)
        spike_out = self.flatten(spike_out)
        spike_out = self.fc1(spike_out)
        spike_out = self.relu7(spike_out)
        spike_out = self.fc2(spike_out)
        spike_out = self.relu8(spike_out)
        spike_out = self.fc3(spike_out)
        spike_out, mem_out = self.relu9(spike_out)
        return spike_out, mem_out
