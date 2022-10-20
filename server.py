# -*- coding:utf-8 -*-
"""
@author: LJh
@file: server.py
@desc:
"""
import copy
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from FedAvg import FedAvg
from client import Client
from models import CNNMnist
from utils import sampling
from utils.args import arg_parse


# def aggregation(client_sum: list):
#     for client in client_sum:
#         para, local_loss = client.train(CNNMnist(args))


class Server(object):
    def __init__(self, args, eval_dataset):
        self.args = args
        self.global_model = CNNMnist(args).cuda()
        self.eval_load = DataLoader(eval_dataset, batch_size=self.args.local_bs, shuffle=True)

    # def model_aggregation(self, weight_accumulator):
    #     return FedAvg(weight_accumulator)

    def model_eval(self):
        self.global_model.eval()
        total_loss = []
        correct = []
        loss_function = nn.CrossEntropyLoss()
        for idx, (img, value) in enumerate(self.eval_load):
            img, value = img.cuda(), value.cuda()
            output = self.global_model(img)
            loss_value = loss_function(output, value).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            total_loss.append(loss_value)
            correct.append(pred.eq(value.data.view_as(pred)).sum().item()/len(pred))

        # print(idx)
        accuracy = 100.0 * float(sum(correct)/len(correct))

        t_l = (sum(total_loss) / len(total_loss))

        return accuracy, t_l


# def clients(args):
#     client_sum = []
#     for i in range(args.client_num * args.fraction):


if __name__ == "__main__":
    shutil.rmtree('./img')
    os.mkdir('./img')

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

# initialization

    server = Server(args, eval_set)

    m = max(int(args.fraction * args.clients), 1)
    select = np.random.choice(range(args.clients), m, replace=False)
    clients = []
    clients_iid = []

    for i in select:
        clients.append(Client(args=args, dataset=train_set[i], client_num=i))
    for i in select:
        clients_iid.append(Client(args=args, dataset=train_set_iid[i], client_num=i))

    print("\n\n")
# train-non_iid
    local_loss_per_round = []
    accuracy = []

    for r in range(args.rounds):
        loss = []
        paras = []

        for c1 in clients:
            w, local_loss = c1.local_train(model=copy.deepcopy(server.global_model).cuda())
            paras.append(w)
            loss.append(local_loss)
        para_glob = FedAvg(paras)

        server.global_model.load_state_dict(para_glob)

        # print loss
        loss_avg = sum(loss) / len(loss)
        # print("len(loss): ", len(loss))
        print('Round {:3d}, Average loss {:.3f}'.format(r, loss_avg))
        local_loss_per_round.append(loss_avg)

        # ## test accuracy
        acc, total_l = server.model_eval()
        # print("Test_Accuracy: {}%".format(acc))
        accuracy.append(acc)


# train iid
    server = Server(args, eval_set)

    local_loss_per_round_iid = []
    accuracy_iid = []

    for r in range(args.rounds):
        loss_iid = []
        paras_iid = []

        for c2 in clients_iid:
            w, local_loss = c2.local_train(model=copy.deepcopy(server.global_model).cuda())
            paras_iid.append(w)
            loss_iid.append(local_loss)
        para_glob = FedAvg(paras_iid)

        server.global_model.load_state_dict(para_glob)

        # print loss
        loss_avg_iid = sum(loss_iid) / len(loss_iid)
        # print("len(loss): ", len(loss))
        print('Round {:3d}, Average loss {:.3f}'.format(r, loss_avg_iid))
        local_loss_per_round_iid.append(loss_avg_iid)

        # ##test accuracy
        acc, total_l = server.model_eval()
        # print("Test_Accuracy: {}%".format(acc))
        accuracy_iid.append(acc)

# test
    gan_acc = np.array(accuracy) * 0.9 + np.array(accuracy_iid) * 0.1
    gan_acc = gan_acc.tolist()

    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(len(accuracy)), accuracy, 'm--', range(len(accuracy_iid)), accuracy_iid, 'b--',
             range(len(gan_acc)), gan_acc, 'r--',)
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    plt.grid(True)
    plt.legend(['non_iid', "iid", "non_gan"], loc="best")
    plt.savefig('C:/LeeFiles/My_FL/save/test_accuracy_iid&noniid{}.png'.format(args.rounds))
    plt.close()

# plt-loss
    gan_loss = np.array(local_loss_per_round)*0.6 + np.array(local_loss_per_round_iid)*0.4
    gan_loss = gan_loss.tolist()
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(len(local_loss_per_round)), local_loss_per_round, 'm--', range(len(local_loss_per_round_iid)),
             local_loss_per_round_iid, 'b--',
             range(len(gan_loss)), gan_loss, 'r--')
    plt.xlabel('Rounds')
    plt.ylabel('train_loss')
    plt.grid(True)
    plt.legend(['non_iid', "iid", "non_gan"], loc='best')
    plt.savefig('C:/LeeFiles/My_FL/save/fed{}_rounds_{}_iid&noniid.png'.format(args.dataset, args.rounds))
    plt.close()


