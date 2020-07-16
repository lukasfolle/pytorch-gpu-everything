import torch
from collections import OrderedDict
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from example.resnet_pytorch import train, test, Net, parse_args
from gpu_everything.augmentation import RandomFlip, RandomRotate90
from gpu_everything.preprocessing import ZScoreNorm, Unsqueeze, Squeeze


def main_traditional():
    args, use_cuda, device, kwargs = parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


def main_gpu_everything():
    args, use_cuda, device, kwargs = parse_args()

    Augmentations = torch.nn.Sequential(OrderedDict([
        ("RandomFlip", RandomFlip(dims=(0, 1), prob=0.5)),
        ("RandomRotate", RandomRotate90(dims=(0, 1), prob=0.8, max_number_rot=3))]))
    Preprocessings = torch.nn.Sequential(OrderedDict([
        ("ZNorm", ZScoreNorm()),
        ("Squeeze", Squeeze()),
        ("Unsqueeze", Unsqueeze(dim=1))
    ]))

    NetGpuEverything = torch.nn.Sequential(OrderedDict([("Augmentations", Augmentations),
                                                        ("Preprocessings", Preprocessings),
                                                        ("Network", Net())]))

    transform = transforms.Compose([transforms.ToTensor()])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = NetGpuEverything.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main_gpu_everything()
    #main_traditional()
