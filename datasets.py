# -*- coding:utf-8 -*-
"""
@author: LJh
@file: datasets.py
@desc:
"""
import numpy as np
import argparse
from torchvision import datasets, transforms

# 设置数据集
pth_mnist = './data/mnist'
pth_fmnist = './data/fmnist'
pth_cifar = './data/cifar10'
trans = transforms.ToTensor
mnist_train = datasets.MNIST(root=pth_mnist, download=True, train=True, transform=trans)
mnist_test = datasets.MNIST(root=pth_mnist, download=True, train=False, transform=trans)

# fmnist_train = datasets.FashionMNIST(root=pth_fmnist, download=True, train=True, transform=trans)
# fmnist_test = datasets.FashionMNIST(root=pth_fmnist, download=True, train=False, transform=trans)
#
# cifar10_train = datasets.CIFAR10(root=pth_cifar, download=True, train=True, transform=trans)
# cifar10_test = datasets.CIFAR10(root=pth_cifar, download=True, train=False, transform=trans)


if __name__ == '__main__':
    print(len(mnist_train))
