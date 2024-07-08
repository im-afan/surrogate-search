import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from models.ds_model import DynamicSurrogateModel
import snntorch as snn
from snntorch.surrogate import FastSigmoid

def train(model: models.ds_model.DynamicSurrogateModel, 
          train_loader: DataLoader, 
          test_loader: DataLoader,
          epochs: int = 3,
          k_entropy: float = 0):

    theta = torch.ones((2,)) # defines mean and logstd

    loss = nn.CrossEntropyLoss()
    model_optim = torch.optim.Adam(net.parameters(), lr=5e-4)
    dist_optim = torch.optim.Adam(net.parameters(), lr=5e-4)
    loss_hist = []
    test_loss_hist = []
    num_epochs = 3

    for epoch in range(num_epochs):
      counter = 0

    total_loss = 0
    prev_loss = 0
    
    for batch_data, batch_labels in train_loader:
        dist = Normal(theta[0], torch.exp(theta[1]))
        temp = dist.sample()

        model.set_surrogate(snn(slope=temp)) # todo: implement dspike

        spikes_out, mem_out = model(batch_data.view(batch_data.size()[0], -1))
        model_loss = torch.zeros(1, device=device, dtype=torch.float)
        for step in range(num_steps):
            model_loss += loss(mem_out[step], batch_labels)
        model_optim.zero_grad()
        model_loss.backward()
        optimizer.step()

        total_loss += model_loss.item()

        loss_change = loss_val.detach() - prev_loss
        dist_loss = (loss_change - dist.entropy()) * dist.log_prob(temp) # max -dloss + entropy => min dloss - entropy
        dist_optim.zero_grad()
        dist_loss.backward()
        dist_optim.step()

        prev_loss = loss_val.detach()



