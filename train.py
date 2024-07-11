import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import numpy as np
import snntorch as snn
from snntorch.surrogate import FastSigmoid
from snntorch import utils
from surrogates import atan_surrogate, tanh_surrogate 
import surrogates
from data import snn_transforms
import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_surrogate(model: nn.Module, surrogate):
    for module in model.modules():
        #if(type(module) == snn.Leaky):
        if(isinstance(module, snn.Leaky)):
            #print("replace lol")
            module.spike_grad = surrogate

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

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

def train(model: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader,
          epochs: int = 3,
          k_entropy: float = 0.1,
          learning_rate: float = 0.01, 
          timesteps: int = 10,
          num_classes: int = 10, 
          use_dynamic_surrogate: bool = True):

    theta = torch.tensor([1, -4], requires_grad=True, device=device, dtype=torch.float32)

    writer = SummaryWriter()
    loss = nn.CrossEntropyLoss()
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dist_optim = torch.optim.SGD([theta], lr=0.001)
    loss_hist = []
    test_loss_hist = []
    #eps = 1-1e-4
    #std = 0.1

    train_steps = 0

    for epoch in range(epochs):
        counter = 0

        total_loss = 0
        prev_loss = 0
        
        for batch_data, batch_labels in train_loader:
            train_steps += 1

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            batch_data = torch.movedim(batch_data, 1, 0) 
            #print(model(batch_data))
            dist = Normal(theta[0], torch.exp(theta[1]))
            #std *= eps
            temp = dist.sample()
            if(not use_dynamic_surrogate):
                temp = torch.tensor(1)
            #set_surrogate(model, surrogates.atan_surrogate(width=0.5)) # todo: implement dspike
            #set_surrogate(model, snn.surrogate.fast_sigmoid(slope=25)) # todo: implement dspike
            #print(temp)
            #set_surrogate(model, surrogates.tanh_surrogate(width=0.5))
            set_surrogate(model, snn.surrogate.custom_surrogate(surrogates.tanh_surrogate1(width=torch.abs(temp))))

            spikes_out, mem_out = forward_pass(model, timesteps, batch_data) 
            model_loss = torch.zeros(1, device=device, dtype=torch.float)
            for step in range(timesteps):
                #print(spikes_out[step].dtype, F.one_hot(batch_labels, num_classes=num_classes).dtype)
                model_loss += loss(mem_out[step], F.one_hot(batch_labels, num_classes=num_classes).to(dtype=torch.float32))

            with torch.autograd.set_detect_anomaly(True): 
                model_optim.zero_grad()
                model_loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                model_optim.step()

            total_loss += model_loss.item()


            loss_change = model_loss.detach() - prev_loss
            if(use_dynamic_surrogate):
                dist_loss = (loss_change - k_entropy * dist.entropy().detach()) * dist.log_prob(temp) # max -dloss + entropy => min dloss - entropy
                dist_optim.zero_grad()
                dist_loss.backward()
                #nn.utils.clip_grad_norm_([theta], 0.01)
                dist_optim.step()

            prev_loss = model_loss.detach()
            if(train_steps % 50 == 0):
                print(f'Loss: {model_loss.item()}, Normal params: {theta[0].item(), theta[1].item()}, temp: {temp.item()}')
            writer.add_scalar("Loss/train", model_loss.item(), train_steps)
            #print(f'Loss: {model_loss.item()}')
        acc = test(model, test_loader, timesteps=timesteps)
        writer.add_scalar("Accuracy/test", acc)
        print(f'Test accuracy after {epoch} epochs: {acc}')
        print(f'Average Loss: {total_loss / len(train_loader)}')
    writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "MNIST"])
    parser.add_argument("--arch", default="resnet18", type=str, choices=["resnet18", "vgg16"])
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--timesteps", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--encoding", default="rate", type=str, choices=["rate", "temporal"])
    parser.add_argument("--use_dynamic_surrogate", default=1, type=int)

    args = parser.parse_args()

    transforms = [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        #v2.RandomResizedCrop(size=(224, 224)),
    ]

    if(args.encoding == "rate"):
        transforms.append(snn_transforms.RateCodeTransform(timesteps=args.timesteps))
    if(args.encoding == "temporal"):
        transforms.append(snn_transforms.TemporalCodeTransform(timesteps=args.timesteps))

    if(args.dataset == "CIFAR10"):
        transform = v2.Compose(transforms)
        num_classes = 10
        train_data = datasets.CIFAR10(root="data/datasets/cifar10", train=True, download=True, transform=transform) 
        test_data = datasets.CIFAR10(root="data/datasets/cifar10", train=False, download=True, transform=transform) 
    if(args.dataset == "CIFAR100"):    
        transform = v2.Compose(transforms)
        num_classes = 100
        train_data = datasets.CIFAR100(root="data/datasets/cifar100", train=True, download=True, transform=transform) 
        test_data = datasets.CIFAR100(root="data/datasets/cifar100", train=False, download=True, transform=transform) 
    if(args.dataset == "MNIST"):
        transforms.insert(-1, snn_transforms.ExpandChannelsTransform())
        transform = v2.Compose(transforms)
        num_classes = 10
        train_data = datasets.MNIST(root="data/datasets/mnist", train=True, download=True, transform=transform) 
        test_data = datasets.MNIST(root="data/datasets/mnist", train=False, download=True, transform=transform) 

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    if(args.arch == "resnet18"):
        model = models.spiking_resnet.resnet18(beta=args.beta, num_classes=num_classes)
        #model = models.spiking_cnn.SpikingCNN()
    if(args.arch == "vgg16"):
        model = models.spiking_vgg.vgg16_bn(beta=args.beta, num_classes=num_classes)
    model = model.to(device)
    train(model, 
          train_loader=train_loader, 
          test_loader=test_loader,
          epochs=args.epochs,
          k_entropy=0,
          learning_rate=args.learning_rate,
          timesteps=args.timesteps,
          use_dynamic_surrogate=args.use_dynamic_surrogate)
