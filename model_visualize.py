from torchviz import make_dot
import hiddenlayer as hl
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import models 
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", type=str, choices=["resnet18", "vgg16", "spikingcnn", "vgg11", "mnistnet"])
    args = parser.parse_args()
    transforms_list_test = [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]

    transform_test = transforms.Compose(transforms_list_test)
    num_classes = 10
    test_data = datasets.CIFAR10(root="data/datasets/cifar10", train=False, download=True, transform=transform_test) 
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    if(args.arch == "resnet18"):
        model = models.resnet.resnet19_cifar(num_classes=num_classes)
        # model = models.spiking_resnet.resnet18(beta=args.beta, num_classes=num_classes)
        # model = models.spiking_cnn.SpikingCNN()
    if(args.arch == "vgg16"):
        model = models.vgg.vgg16_bn(num_classes=num_classes)
        # model = models.spiking_vgg.vgg16_bn(beta=args.beta, num_classes=num_classes)
    if(args.arch == "vgg11"):
        model = models.vgg.vgg11_bn(num_classes=num_classes)
    if(args.arch == 'spikingcnn'):
        model = models.conv.SimpleCNN()
        # model = models.spiking_cnn_deep.SpikingCNNDeep()
    if(args.arch == 'mnistnet'):
        model = models.mnistnet.MNISTNet()

    #model = models.to_spiking.SNN(model)
    print(model)
    transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.

    data, label = next(iter(test_loader))
    y = model(data)

    torch.onnx.export(model, data, 'vgg16.onnx', input_names=['image'], output_names=['pred'])


