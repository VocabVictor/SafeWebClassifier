# -*- coding: utf-8 -*-
# !/usr/bin/python3
# @Time    : 2023/1/31 15:00
# @Author  : VocabVictor
# @Email   : VocabVictor@gmail.com
# @File    : base_data_loader.py
# @Software: PyCharm,VsCode
# @Description: 自己手写的一个pytorch的DataLoader，可以流水线输出独热码
# @Support Python Version: 3.5+

from typing import Tuple, Iterator, Union, TypeVar
from torch.utils.data.dataset import T_co
from torchdata.datapipes.iter import IterDataPipe
from polars import read_csv

D = TypeVar("D")


class OneHot(IterDataPipe):

    def __getitem__(self, index) -> T_co:
        pass

    def __init__(
            self,
            source_datapipe,
            num_class=-1,
            **fmtparams,
    ) -> None:
        self.source_datapipe = source_datapipe
        self.fmtparams = fmtparams
        self.num_class = num_class

    def __iter__(self) -> Iterator[Union[D, Tuple[str, D]]]:
        for path, file in self.source_datapipe:
            stream = self.read_file(file)
            yield from self._return_path(path, stream)
