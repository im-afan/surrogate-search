import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import snntorch as snn
from snntorch import surrogate

class Cifar10SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        spike_grad = surrogate.leaky(beta=0.9)

        # Define layers
        self.conv1 = snn.SpikeConv2d(3, 32, kernel_size=3, padding=1, surrogate_function=spike_grad)
        self.relu1 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.conv2 = snn.SpikeConv2d(32, 64, kernel_size=3, stride=1, padding=1, surrogate_function=spike_grad)
        self.relu2 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.pool1 = snn.SpikeAvgPool2d(2, 2)
        self.conv3 = snn.SpikeConv2d(64, 128, kernel_size=3, stride=1, padding=1, surrogate_function=spike_grad)
        self.relu3 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.conv4 = snn.SpikeConv2d(128, 128, kernel_size=3, stride=1, padding=1, surrogate_function=spike_grad)
        self.relu4 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.pool2 = snn.SpikeAvgPool2d(2, 2)
        self.conv5 = snn.SpikeConv2d(128, 256, kernel_size=3, stride=1, padding=1, surrogate_function=spike_grad)
        self.relu5 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.conv6 = snn.SpikeConv2d(256, 256, kernel_size=3, stride=1, padding=1, surrogate_function=spike_grad)
        self.relu6 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.pool3 = snn.SpikeAvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = snn.SpikeLinear(256 * 4 * 4, 1024, surrogate_function=spike_grad)
        self.relu7 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.fc2 = snn.SpikeLinear(1024, 512, surrogate_function=spike_grad)
        self.relu8 = snn.Leaky(beta=0.5, surrogate_function=spike_grad)
        self.fc3 = snn.SpikeLinear(512, 10, surrogate_function=spike_grad)

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
        return spike_out
