# # -*- coding:utf-8 -*-
# """
# @author: LJh
# @file: test.py
# @desc:
# """
import copy
from utils.sampling import fmnist_noniid
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import datasets, transforms
from models import CNNMnist
from client import MyDataset
import server
#
# pth_mnist = './data/mnist'
# pth_fmnist = './data/fmnist'
# pth_cifar = './data/cifar10'
# trans = transforms.ToTensor()
#
# fmnist_train = datasets.FashionMNIST(root=pth_fmnist, download=False, train=True, transform=trans)
# fmnist_test = datasets.FashionMNIST(root=pth_fmnist, download=False, train=False, transform=trans)
#
# cifar10_train = datasets.CIFAR10(root=pth_cifar, download=False, train=True, transform=trans)
# cifar10_test = datasets.CIFAR10(root=pth_cifar, download=False, train=False, transform=trans)
#
#
# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label
#
#
# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset) / num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
#
#
# train_set = mnist_iid(fmnist_train, 100)
#
# ldr_train = DataLoader(DatasetSplit(fmnist_train, train_set[0]), batch_size=10, shuffle=True)
#
# import torch
# from models import CNNMnist
# from utils.args import arg_parse
#
# args = arg_parse()
# cnn = CNNMnist(args)
#
# labels = fmnist_train.targets
# pth_mnist = './data/mnist'
# pth_fmnist = './data/fmnist'
# pth_cifar = './data/cifar10'
# trans = transforms.ToTensor()
#
# fmnist_train = datasets.FashionMNIST(root=pth_fmnist, download=False, train=True, transform=trans)
# fmnist_test = datasets.FashionMNIST(root=pth_fmnist, download=False, train=False, transform=trans)
#
# index = labels.argsort()
# dataset1 = copy.deepcopy(fmnist_train)
# dataset1.data, dataset1.targets = dataset1.data[index], dataset1.targets[index]
# from torch.utils.data import Subset
#
#
# indices = list(range(100, 500)) + list(range(600, 800))
# print(indices)
# # new_sort = Subset(dataset1, indices)
# # print(len(new_sort))
# # n = DataLoader(new_sort, batch_size=10, shuffle=True)
# # for i, j in n:
# #     print(i)
# #     print(j)
# # for i, j in new_sort:
# #     print(i.shape)
# # print('--------------------------------------')
# # D = split_fmnist(fmnist_train, args)
# # print(len(D[0]))
# # local_trainset = DataLoader(MyDataset(D[0]), batch_size=args.local_bs, shuffle=False)
# # for i, j in local_trainset:
# #     print(i)

# input = torch.randn(3, 5, requires_grad=True)
#
# target = torch.randn(3, 5).softmax(dim=1)
# print(input, '\n', target)
# loss = nn.CrossEntropyLoss()(input, target)
# # loss(input, target)
# print(loss)

from utils.args import arg_parse
from utils import sampling

if __name__ == "__main__":

    print("Start Training.............")
    args = arg_parse()

    pth_mnist = './data/mnist'
    pth_fmnist = './data/fmnist'
    pth_cifar = './data/cifar10'
    trans = transforms.ToTensor()

    mnist_train = datasets.MNIST(root=pth_mnist, download=True, train=True, transform=trans)
    mnist_test = datasets.MNIST(root=pth_mnist, download=True, train=False, transform=trans)

    fmnist_train = datasets.FashionMNIST(root=pth_fmnist, download=False, train=True, transform=trans)
    fmnist_test = datasets.FashionMNIST(root=pth_fmnist, download=False, train=False, transform=trans)

# datasets setting

    train_set = sampling.fmnist_noniid(dataset=fmnist_train, args=args)
    train_set_iid = sampling.fmnist_iid(dataset=fmnist_train, args=args)
    eval_set = fmnist_test

# train-main

    server = server.Server(args, eval_set)
    w = server.global_model.state_dict()
    para = copy.deepcopy(w)
    print(para)
    print(para.keys())
    print(len(para))
    # for k in para.keys():
    #     for i in range(1, len(para)):
    #         print(para[k])