# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 数据加载器(data_loader)
# @Support Python Version: 3.5+

from torchdata.datapipes.iter import FileLister
from Base import BaseDataLoader
from Config import Config
from DataLoader.DataPipe.textreader import CsvReader
from re import sub


class DataLoader(BaseDataLoader):

    def __init__(self, test=False):
        if test:
            config = Config().test_loader.args
            datapipe = FileLister(config.data_dir, "*.csv")
        else:
            config = Config().train_loader.args
            datapipe = FileLister(config.data_dir, "*.csv")

        self.label = {}
        self.get_label(datapipe)
        datapipe = datapipe.open_files(mode="r")
        datapipe = CsvReader(datapipe, return_path=True)
        super().__init__(datapipe, self.label, test)

    def get_label(self, datapipe):
        for name in datapipe:
            index = int(sub(r"\D*(\d+)\D*", r"\1", name))
            index = index // 45
            self.label[name] = index


if __name__ == "__main__":
    for data, target in DataLoader():
        print(data, target)
