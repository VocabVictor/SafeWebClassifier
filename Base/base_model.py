# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 模型基类
# @Support Python Version: 3.5+

import torch.nn as nn
from abc import abstractmethod
from numpy import prod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
