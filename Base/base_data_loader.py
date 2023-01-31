# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 数据加载器(data_loader)基类
# @Support Python Version: 3.5+

"""
    这段代码实现了一个基础数据加载器(BaseDataLoader)类，继承自torchdata.dataloader2中的DataLoader2类。
    BaseDataLoader类中实现了数据加载的一些操作，包括打乱数据、分离标签数据和训练数据、按照指定批量大小分批、将数据与标签进行合并。
    同时，BaseDataLoader类也实现了一个拼接数据的函数，并通过重载len()函数实现了对数据长度的统计。
"""

from torch import tensor
from torchdata.dataloader2 import DataLoader2
from Config import Config


class BaseDataLoader(DataLoader2):
    __length = 0  # 初始化数据长度为0

    def init(self,
             datapipe,
             label=None,
             test=False,
             unzip=True,
             datapipe_adapter_fn=None,
             reading_service=None
             ):

        datapipe = datapipe.shuffle()  # 打乱数据

        if test:
            config = Config().test_loader.args  # 如果为测试模式，使用测试模式的配置参数
        else:
            config = Config().train_loader.args  # 否则使用训练模式的配置参数

        if unzip:
            labelpipe, datapipe = datapipe.unzip(sequence_length=2)  # 分离数据和标签
            labelpipe = labelpipe.map(lambda x: label[x])  # 将标签索引映射为对应的标签数据

        self.batch_size = config.batch_size  # 获取批量大小

        datapipe = datapipe.batch(batch_size=config.batch_size, drop_last=config.drop_last)  # 对数据进行分批
        labelpipe = labelpipe.batch(batch_size=config.batch_size, drop_last=config.drop_last)  # 对标签进行分批

        datapipe = datapipe.collate(self.collate_fn)  # 对数据进行拼接
        labelpipe = labelpipe.collate(self.collate_fn)  # 对标签进行拼接

        datapipe = datapipe.zip(labelpipe)  # 将数据和标签进行合并

        # 调用父类的init函数
        super().init(
            datapipe,
            datapipe_adapter_fn,
            reading_service
        )

    def collate_fn(self, batch):
        new_batch = tensor(batch)
        self.__length += len(new_batch)
        return new_batch

    def __len__(self):
        return self.__length // 2
