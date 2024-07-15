# -*- coding: utf-8 -*-
"""SRA Project 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15BgZmwL8hHjF4ahB635MBW55VNJd1tD_
"""

#Libraries

import torch
import torch.nn as nn
import snntorch as snn
from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, beta: float = 0.5
    ) -> None:
        super().__init__()
        self.beta = beta
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            snn.Leaky(beta=beta), #change here
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            snn.Leaky(beta=beta), #change here
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
            snn.Leaky(beta=beta)
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, num_steps: int=100) -> torch.Tensor: #change here
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, beta: float = 0.5) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), snn.Leaky(beta=beta)]
            else:
                layers += [conv2d, snn.Leaky(beta=beta)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

def _vgg(cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg16(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", False, progress, **kwargs)

def vgg16_bn(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", True, progress, **kwargs)