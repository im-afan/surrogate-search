import sys
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
from torchvision import transforms
import numpy as np
import snntorch as snn
from snntorch.surrogate import FastSigmoid
from snntorch import utils
from models.to_spiking import to_spiking
from surrogates import atan_surrogate, tanh_surrogate, dspike1, tanh_surrogate1
import surrogates
from data import snn_transforms
import models
import time
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torchvision.models import resnet18, vgg16_bn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_surrogate(module: nn.Module, surrogate):
    for name, child_module in module.named_children():
        #if(type(module) == snn.Leaky):
        if(isinstance(child_module, nn.Sequential)):
            set_surrogate(child_module, surrogate)
        if(isinstance(child_module, snn.Leaky)):
            #print("replace lol")
            setattr(child_module, "spike_grad", surrogate)
            #module.spike_grad = surrogate

def forward_pass(net, num_steps, data):
  #mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      #spk_out, mem_out = net(data[step])
      spk_out = net(data[step])
      spk_rec.append(spk_out)
      #mem_rec.append(mem_out)

  return torch.stack(spk_rec)#, torch.stack(mem_rec)

def test(model: nn.Module, test_loader: DataLoader, timesteps: int = 10):
  total = 0
  correct = 0
  for batch_data, batch_labels in test_loader:
    batch_data = batch_data.to(device)
    batch_labels = batch_labels.to(device)

    batch_data = torch.movedim(batch_data, 1, 0) 
    spikes_out = forward_pass(model, timesteps, batch_data) 
    pred = spikes_out.sum(dim=0).argmax(1)
    total += len(batch_labels)
    correct += (pred == batch_labels).detach().cpu().sum().numpy()
  return correct / total

def train_categorical(
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
      #print(batch_data.shape)

      #sample temperature from categorical distribution 
      probs = F.softmax(logits, dim=0)
      dist = Categorical(probs)
      temp_idx = dist.sample()
      temp = torch.tensor(candidate_temps[temp_idx], device=device, dtype=torch.float32)
      
      #set_surrogate(model, snn.surrogate.custom_surrogate(dspike(b=torch.abs(temp))))
      set_surrogate(model, snn.surrogate.custom_surrogate(surrogates.dspike1(b=0.5)))


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



def train(model: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader,
          epochs: int = 3,
          k_entropy: float = 0.1,
          model_learning_rate: float = 0.01, 
          dist_learning_rate: float = 0.001,
          timesteps: int = 10,
          num_classes: int = 10, 
          use_dynamic_surrogate: bool = True,
          mean: float = 1,
          logstd: float = -4):

    theta = torch.tensor([mean, logstd], requires_grad=True, device=device, dtype=torch.float32)

    writer = SummaryWriter()
    loss = nn.CrossEntropyLoss()
    
    model_optim = torch.optim.SGD(model.parameters(), lr=model_learning_rate, momentum=0.9, weight_decay=5e-4)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, eta_min=0, T_max=epochs)
    dist_optim = torch.optim.SGD([theta], lr=dist_learning_rate)
    #dist_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dist_optim, eta_min=0, T_max=epochs)


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

            #print(batch_data)
            batch_data = torch.movedim(batch_data, 1, 0) 
            #print(model(batch_data))
            dist = Normal(theta[0], torch.exp(theta[1]))
            #std *= eps
            temp = dist.sample()
            if(not use_dynamic_surrogate):
                temp = torch.tensor(mean)
            #set_surrogate(model, surrogates.atan_surrogate(width=0.5)) # todo: implement dspike
            #set_surrogate(model, snn.surrogate.fast_sigmoid(slope=25)) # todo: implement dspike
            #print(temp)
            #set_surrogate(model, surrogates.tanh_surrogate(width=0.5))
            set_surrogate(model, snn.surrogate.custom_surrogate(surrogates.tanh_surrogate1(width=torch.abs(temp))))

            #spikes_out, mem_out = forward_pass(model, timesteps, batch_data) 
            spikes_out = forward_pass(model, timesteps, batch_data) 
            model_loss = torch.zeros(1, device=device, dtype=torch.float)
            for step in range(timesteps):
                #print(spikes_out[step].dtype, F.one_hot(batch_labels, num_classes=num_classes).dtype)
                #print(spikes_out.shape)
                model_loss += loss(spikes_out[step], F.one_hot(batch_labels, num_classes=num_classes).to(dtype=torch.float32))

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
        format_string = '%Y-%m-%d_%H:%M:%S'
        cur_time = time.strftime(format_string, time.gmtime())
        if(use_dynamic_surrogate):
            torch.save(model.state_dict(), "runs/saves/dynamic_surrogate_" + cur_time + ".pt")
        else:
            torch.save(model.state_dict(), "runs/saves/static_surrogate_" + cur_time + ".pt")
        acc = test(model, test_loader, timesteps=timesteps)
        writer.add_scalar("Accuracy/test", acc)
        model_scheduler.step()
        #dist_scheduler.step()
        print(f'Test accuracy after {epoch} epochs: {acc}')
        print(f'Average Loss: {total_loss / len(train_loader)}')
    writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100", "MNIST"])
    parser.add_argument("--arch", default="resnet18", type=str, choices=["resnet18", "vgg16", "spikingcnn"])
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--model_learning_rate", default=1e-3, type=float)
    parser.add_argument("--dist_learning_rate", default=1e-3, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--timesteps", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--encoding", default="rate", type=str, choices=["rate", "temporal"])
    parser.add_argument("--use_dynamic_surrogate", default=1, type=int)
    parser.add_argument("--initial_temp", default=1, type=float)
    parser.add_argument("--initial_logstd", default=-4, type=float)
    parser.add_argument("--training_type", default="train", type=str, choices=["train","train_categorical"])

    args = parser.parse_args()

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        #transforms.RandomResizedCrop(size=(224, 224)),
    ]

    if(args.dataset == "CIFAR10" or args.dataset == "CIFAR100"):
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    if(args.encoding == "rate"):
        transforms_list.append(snn_transforms.RateCodeTransform(timesteps=args.timesteps))
    if(args.encoding == "temporal"):
        transforms_list.append(snn_transforms.TemporalCodeTransform(timesteps=args.timesteps))

    

    if(args.dataset == "CIFAR10"):
        transform = transforms.Compose(transforms_list)
        num_classes = 10
        train_data = datasets.CIFAR10(root="data/datasets/cifar10", train=True, download=True, transform=transform) 
        test_data = datasets.CIFAR10(root="data/datasets/cifar10", train=False, download=True, transform=transform) 
    if(args.dataset == "CIFAR100"):    
        transform = transforms.Compose(transforms_list)
        num_classes = 100
        train_data = datasets.CIFAR100(root="data/datasets/cifar100", train=True, download=True, transform=transform) 
        test_data = datasets.CIFAR100(root="data/datasets/cifar100", train=False, download=True, transform=transform) 
    if(args.dataset == "MNIST"):
        transforms_list.insert(-1, snn_transforms.ExpandChannelsTransform())
        transform = transforms.Compose(transforms_list)
        num_classes = 10
        train_data = datasets.MNIST(root="data/datasets/mnist", train=True, download=True, transform=transform) 
        test_data = datasets.MNIST(root="data/datasets/mnist", train=False, download=True, transform=transform) 

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    """
    batch, labels = next(iter(train_loader))
    print(batch.shape)
    img = torchvision.utils.make_grid(batch)
    img = torch.movedim(img, 0, 2)
    print(img.shape)
    print(img)
    plt.imshow(img)
    plt.show()
    """

    if(args.arch == "resnet18"):
        model = resnet18(num_classes=num_classes)
        #model = models.spiking_resnet.resnet18(beta=args.beta, num_classes=num_classes)
        #model = models.spiking_cnn.SpikingCNN()
    if(args.arch == "vgg16"):
        model = vgg16_bn(num_classes=num_classes)
        #model = models.spiking_vgg.vgg16_bn(beta=args.beta, num_classes=num_classes)
    if(args.arch == 'spikingcnn'):
        model = models.spiking_cnn.SpikingCNN()
        #model = models.spiking_cnn_deep.SpikingCNNDeep()

    to_spiking(model) 
    model = model.to(device)

    if(args.training_type == "train"): 
        train(model, 
              train_loader=train_loader, 
              test_loader=test_loader,
              epochs=args.epochs,
              k_entropy=0,
              model_learning_rate=args.model_learning_rate,
              dist_learning_rate=args.dist_learning_rate,
              timesteps=args.timesteps,
              use_dynamic_surrogate=args.use_dynamic_surrogate,
              mean=args.initial_temp,
              logstd=args.initial_logstd)

    if(args.training_type == "train_categorical"): 
        candidate_temps = [5.0,10.0,15.0,25.0]
        train_categorical(model, 
        train_loader=train_loader, 
        test_loader=test_loader,
        epochs=3,
        k_entropy=0,
        learning_rate= 1e-3,
        timesteps=10,
        candidate_temps=candidate_temps
    )
