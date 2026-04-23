import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, Dataset

##Setting Train Datasets

# CIFAR-10
cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# SVHN
svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
svhn_test  = datasets.SVHN(root='./data', split='test', download=True, transform=transform)