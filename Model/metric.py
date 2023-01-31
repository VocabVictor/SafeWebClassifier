# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 封装了常用的metric(训练指标)计算方法
# @Support Python Version: 3.5+

import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
