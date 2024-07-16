import sys
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import snntorch as snn
from snntorch.surrogate import FastSigmoid
from snntorch import utils
import time
from spikingjelly.clock_driven import functional, surrogate, neuron
import spikingjelly.activation_based.model as spiking_resnet
from data import snn_transforms
from spikingjelly.activation_based import neuron, functional, surrogate
from spikingjelly.activation_based.model import spiking_resnet
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from copy import deepcopy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_surrogate(module: nn.Module, surrogate_func):
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Sequential):
            set_surrogate(child_module, surrogate_func)
        if isinstance(child_module, neuron.IFNode):
            setattr(child_module, "surrogate_function", surrogate_func)

def forward_pass_spikingjelly(net, num_steps, data):
    spk_rec = []
    functional.reset_net(net)
    for step in range(num_steps):
        spk_out = net(data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)

def test_spikingjelly(model: nn.Module, test_loader: DataLoader, timesteps: int = 10):
    total = 0
    correct = 0
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        batch_data = torch.movedim(batch_data, 1, 0)
        spikes_out = forward_pass_spikingjelly(model, timesteps, batch_data)
        pred = spikes_out.sum(dim=0).argmax(1)
        total += len(batch_labels)
        correct += (pred == batch_labels).detach().cpu().sum().numpy()
    return correct / total

def train_spikingjelly(
        model: nn.Module,
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

    train_steps = 0

    for epoch in range(epochs):
        total_loss = 0
        prev_loss = 0

        for batch_data, batch_labels in train_loader:
            train_steps += 1

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            batch_data = torch.movedim(batch_data, 1, 0)
            dist = Normal(theta[0], torch.exp(theta[1]))
            temp = dist.sample()
            if not use_dynamic_surrogate:
                temp = torch.tensor(mean)
            set_surrogate(model, surrogate.Sigmoid(torch.abs(temp)))

            spikes_out = forward_pass_spikingjelly(model, timesteps, batch_data)
            model_loss = torch.zeros(1, device=device, dtype=torch.float)
            for step in range(timesteps):
                model_loss += loss(spikes_out[step], batch_labels)

            with torch.autograd.set_detect_anomaly(True):
                model_optim.zero_grad()
                model_loss.backward()
                model_optim.step()

            total_loss += model_loss.item()

            loss_change = model_loss.detach() - prev_loss
            if use_dynamic_surrogate:
                dist_loss = (loss_change - k_entropy * dist.entropy().detach()) * dist.log_prob(temp)
                dist_optim.zero_grad()
                dist_loss.backward()
                dist_optim.step()

            prev_loss = model_loss.detach()
            if train_steps % 50 == 0:
                print(f'Loss: {model_loss.item()}, Normal params: {theta[0].item(), theta[1].item()}, temp: {temp.item()}')
            writer.add_scalar("Loss/train", model_loss.item(), train_steps)

        format_string = '%Y-%m-%d_%H:%M:%S'
        cur_time = time.strftime(format_string, time.gmtime())
        torch.save(model.state_dict(), "runs/saves/spikingjelly_" + cur_time + ".pt")
        acc = test_spikingjelly(model, test_loader, timesteps=timesteps)
        writer.add_scalar("Accuracy/test", acc)
        model_scheduler.step()
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
    parser.add_argument("--training_type", default="train_spikingjelly", type=str, choices=["train", "train_categorical", "train_spikingjelly"])

    args = parser.parse_args()

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    transforms_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]

    if args.dataset in ["CIFAR10", "CIFAR100"]:
        transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    if args.encoding == "rate":
        transforms_list.append(snn_transforms.RateCodeTransform(timesteps=args.timesteps))
    if args.encoding == "temporal":
        transforms_list.append(snn.TemporalCodeTransform(timesteps=args.timesteps))

    if args.dataset == "CIFAR10":
        transform = transforms.Compose(transforms_list)
        num_classes = 10
        train_data = datasets.CIFAR10(root="data/datasets/cifar10", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data/datasets/cifar10", train=False, download=True, transform=transform)
    elif args.dataset == "CIFAR100":
        transform = transforms.Compose(transforms_list)
        num_classes = 100
        train_data = datasets.CIFAR100(root="data/datasets/cifar100", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root="data/datasets/cifar100", train=False, download=True, transform=transform)
    elif args.dataset == "MNIST":
        transforms_list.insert(-1, transforms.Grayscale(num_output_channels=1))
        transform = transforms.Compose(transforms_list)
        num_classes = 10
        train_data = datasets.MNIST(root="data/datasets/mnist", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="data/datasets/mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    if args.arch == "resnet18":
        model = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    elif args.arch == "vgg16":
        model = spiking_resnet.spiking_vgg16(num_classes=num_classes)
    elif args.arch == "spikingcnn":
        model = snn.SpikingCNN(width=32, height=32, channels=3, kernel_size=5, num_classes=num_classes, timesteps=args.timesteps, beta=args.beta)

    model = model.to(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.training_type == "train_spikingjelly":
        train_spikingjelly(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            model_learning_rate=args.model_learning_rate,
            dist_learning_rate=args.dist_learning_rate,
            timesteps=args.timesteps,
            num_classes=num_classes,
            use_dynamic_surrogate=args.use_dynamic_surrogate,
            mean=args.initial_temp,
            logstd=args.initial_logstd
        )
