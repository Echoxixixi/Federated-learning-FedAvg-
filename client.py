# -*- coding:utf-8 -*-
"""
@author: LJh
@file: client.py
@desc:
"""
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from models import CNNMnist
from utils.args import arg_parse
from utils.sampling import fmnist_noniid, fmnist_iid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        img, value = self.dataset[item]
        return img, value

    def __len__(self):
        return len(self.dataset)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Client(object):
    def __init__(self, args, dataset, client_num):
        self.model = CNNMnist(args).to(args.device)
        # self.dataset = dataset
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.selected_clients = []
        self.client_num = client_num
        self.local_trainset = DataLoader(MyDataset(dataset), batch_size=self.args.local_bs, shuffle=False)

    def local_train(self, model):
        model.train()
        epoch_loss = []
        correct = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        for epoch in tqdm(range(self.args.local_ep)):
            batch_loss = []

            for batch_idx, (features, labels) in enumerate(self.local_trainset):
                # features = features.to(torch.float)

                features, labels = features.cuda(), labels.cuda()
                model.zero_grad()
                y_hat = model(features)
                # pred = y_hat.max(1)[1]
                # correct.append(float(pred.eq(labels).sum().item()) / len(pred))
                loss = self.loss_func(y_hat, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append((sum(batch_loss) / len(batch_loss)))
            # print("client{}_acc".format(self.client_num), sum(correct)/len(correct))
            # print(sum(batch_loss), len(batch_loss))

            # print('Epoch{}_loss:{}'.format(epoch, epoch_loss[epoch]))

        # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(range(len(epoch_loss)), epoch_loss, label='local_loss_client{}'.format(self.client_num))
        # # Plot some data on the axes.
        # # ax.plot(x, x ** 2, label='quadratic')  # Plot more data on the axes...
        # # ax.plot(x, x ** 3, label='cubic')  # ... and some more.
        # ax.set_xlabel('epochs')  # Add an x-label to the axes.
        # ax.set_ylabel('loss')  # Add a y-label to the axes.
        # ax.set_title("client{}_loss".format(self.client_num))  # Add a title to the axes.
        # ax.legend()  # Add a legend.
        # fig.savefig('./img/pic-{}.png'.format(self.client_num))
        # plt.close()

        final_loss = sum(epoch_loss) / len(epoch_loss)
        return model.state_dict(), final_loss


if __name__ == '__main__':
    pth_fmnist = './data/fmnist'
    pth_cifar = './data/cifar10'
    trans = transforms.ToTensor()

    fmnist_train = datasets.FashionMNIST(root=pth_fmnist, download=False, train=True, transform=trans)
    fmnist_test = datasets.FashionMNIST(root=pth_fmnist, download=False, train=False, transform=trans)
    args = arg_parse()

    Dtr = fmnist_iid(fmnist_train, args)
    c0 = Client(args=args, dataset=Dtr[0], client_num=0)
    a, b = c0.local_train(CNNMnist(args).to(args.device))
    print("client{}_loss".format(c0.client_num), b)
