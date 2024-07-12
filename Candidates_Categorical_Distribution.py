import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils


import torch
import torch.nn as nn
import torchvision 
from torchvision.transforms import v2 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F 
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import numpy as np
import itertools

import argparse 

import snn_transforms
import models 
from Dspike import dspike, atan_surrogate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_surrogate(model: nn.Module, surrogate): 
  for module in model.modules(): 
    if(type(module)==snn.Leaky): 
      module.spike_grad = surrogate 


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  

    for step in range(num_steps):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def test(model: nn.Module, test_loader: DataLoader, timesteps: int = 10):
    total = 0
    correct = 0
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        batch_data = torch.movedim(batch_data, 1, 0)
        spikes_out, mem_out = forward_pass(model, timesteps, batch_data)
        pred = spikes_out.sum(dim=0).argmax(1)
        total += len(batch_labels)
        correct += (pred == batch_labels).detach().cpu().sum().numpy()
    return correct / total


def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader,
    epochs: int = 3,
    k_entropy: float = 0.01,
    learning_rate: float = 0.01, 
    timesteps: int = 10,
    num_classes: int = 10,
    candidate_temps=[0.5, 1.0, 1.5, 2.0]
): 
  
  
  logits = torch.ones(len(candidate_temps), requires_grad=True, device=device, dtype = torch.float32)

  loss = nn.CrossEntropyLoss()
  model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
  dist_optim = torch.optim.Adam([logits], lr=0.1)
  loss_hist = []
  test_loss_hist = []

  for epoch in range(epochs): 
    counter = 0 
    total_loss = 0
    prev_loss = 0

    for batch_data, batch_labels in train_loader: 
      batch_data = batch_data.to(device)
      batch_labels = batch_labels.to(device)

      batch_data = torch.movedim(batch_data, 1,0)
      print(batch_data.shape)

      #sample temperature from categorical distribution 
      probs = F.softmax(logits, dim=0)
      dist = Categorical(probs)
      temp_idx = dist.sample()
      temp = torch.tensor(candidate_temps[temp_idx], device=device, dtype=torch.float32)
      
      #set_surrogate(model, snn.surrogate.custom_surrogate(dspike(b=torch.abs(temp))))
      set_surrogate(model, snn.surrogate.custom_surrogate(atan_surrogate(width=0.5)))


      spikes_out, mem_out = forward_pass(model, timesteps, batch_data)
      model_loss = torch.zeros(1, device=device, dtype=torch.float)
      for step in range(timesteps):
          model_loss += loss(mem_out[step], F.one_hot(batch_labels, num_classes=num_classes).to(dtype=torch.float32))
      model_optim.zero_grad()
      model_loss.backward()
      model_optim.step()

      total_loss += model_loss.item()

      loss_change = model_loss.detach() - prev_loss
      dist_loss = (loss_change - k_entropy * dist.entropy().detach()) * dist.log_prob(temp_idx)
      dist_optim.zero_grad()
      dist_loss.backward()
      dist_optim.step()

      prev_loss = model_loss.detach()
      print(f'Loss: {model_loss.item()}, Categorical params: {logits}, change loss: {loss_change}, entropy: {dist.entropy().detach()}')
    
    print(f'Test accuracy after {epoch} epochs: {test(model, test_loader, timesteps=timesteps)}')
    print(f'Average Loss: {total_loss / len(train_loader)}')


#downloading data 

batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float

transforms = [
	v2.PILToTensor(), 
	v2.ToDtype(torch.float32), 

]


transforms.append(snn_transforms.RateCodeTransform(timesteps=10))
transforms.insert(-1, snn_transforms.ExpandChannelsTransform())
transform = v2.Compose(transforms)

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)



train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


#establishing the models 

#parser = argparse.ArgumentParser()

#args = parser.parse_args()




#training 


candidate_temps = [0.5,1.0,1.5,2.0]

num_classes = 10

model = models.spiking_resnet.resnet18(beta=0.5, num_classes=10)

#transform = v2.Compose(transforms)

train(model, 
    train_loader=train_loader, 
    test_loader=test_loader,
    epochs=3,
    k_entropy=0,
    learning_rate= 1e-3,
    timesteps=10,
    candidate_temps=candidate_temps
)




#/Users/chetant/Sublime/spiking_resnet.py
















