from typing import Callable
import torchvision
from torchvision.transforms import v2
from snntorch import spikegen

class RateCodeTransform(Callable):
    def __init__(self,
                 timesteps: int = 10,
                 gain: float = 1):
        self.timesteps = timesteps
        self.gain = gain

    def __call__(self, img):
        return spikegen.rate(img, num_steps=self.timesteps, gain=self.gain)

class TemporalCodeTransform(Callable):
    def __init__(self,
                 timesteps: int = 10,
                 threshold: int = 0.01,
                 normalize: bool = True,
                 linear: bool = False):
        self.timesteps = timesteps
        self.threshold = threshold
        self.normalize = normalize
        self.linear = linear

    def __call__(self, img):
        return spikegen.latency(img, 
                                num_steps=self.timesteps, 
                                threshold=self.threshold, 
                                normalize=self.normalize, 
                                linear=self.linear)

class ExpandChannelsTransform(Callable):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.repeat(3, 1, 1)