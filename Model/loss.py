# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 封装了常用的loss计算方法
# @Support Python Version: 3.5+

import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)
