# -*- coding:utf-8 -*-
"""
@author: LJh
@file: FedAvg.py
@desc:
"""
import copy
import torch


def FedAvg(para):
    para_avg = copy.deepcopy(para[0])
    for k in para_avg.keys():
        for i in range(1, len(para)):
            para_avg[k] += para[i][k]
        para_avg[k] = torch.div(para_avg[k], len(para))
    return para_avg


if __name__ == '__main__':
    pass
