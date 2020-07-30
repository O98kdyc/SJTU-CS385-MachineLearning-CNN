import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

def obtain_loader(train_batch=10, test_batch=10):
    train_dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False)
    return train_loader, test_loader