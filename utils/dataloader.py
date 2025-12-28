import torchvision
import torch
from torch.utils.data import DataLoader


def cifar_10_dataloader():
    train_dataset = torchvision.datasets.CIFAR10(root = './cifar10_data',train=True, transform=None, target_transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root = './cifar10_data',train=False, transform=None, target_transform=None, download=True)

    print(f" \nthe class are   {'--'.join(train_dataset.classes)}\n")

    print(f"{train_dataset.data.shape=}")
    print(f"{test_dataset.data.shape=}")
    train_loader = DataLoader(dataset = train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset = test_dataset, batch_size=16, shuffle=True, pin_memory=True)

    return train_dataset, test_dataset, train_loader, test_loader


import matplotlib.pyplot as plt
def view_sample_images(train_dataset):
    nrow, ncol = 4,8
    fig,axes = plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True)
    for i in range(nrow):
        for j in range(ncol):
            axes[i][j].imshow(train_dataset.data[i*ncol+j])


