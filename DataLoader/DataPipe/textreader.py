import abc
from typing import Tuple, IO, Iterator, Union, TypeVar
from torch.utils.data.dataset import T_co
from torchdata.datapipes.iter import IterDataPipe
from polars import read_csv

D = TypeVar("D")


class TextReader(IterDataPipe):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(
            self,
            source_datapipe: IterDataPipe[Tuple[str, IO]],
            *,
            skip_lines: int = 0,
            encoding: str = "utf-8",
            errors: bool = False,
            return_path: bool = False,
            **fmtparams,
    ) -> None:
        self.__length = 0
        self.source_datapipe = source_datapipe
        self.fmtparams = fmtparams
        self.skip_lines = skip_lines
        self.encoding = encoding
        self.errors = errors
        self.return_path = return_path

    def _return_path(self, path, datas):
        if self.return_path:
            for data in datas:
                yield path, data
        else:
            yield from datas

    def __iter__(self) -> Iterator[Union[D, Tuple[str, D]]]:
        for path, file in self.source_datapipe:
            stream = self.read_file(file)
            yield from self._return_path(path, stream)

    @abc.abstractmethod
    def read_file(self, file):
        raise NotImplementedError


class CsvReader(TextReader):

    def read_file(self, file):
        yield from read_csv(
            file,
            has_header=False,
            skip_rows=self.skip_lines,
            encoding=self.encoding,
            ignore_errors=self.errors,
            **self.fmtparams
        ).rows()


class XmlReader(TextReader):

    def read_file(self, file):
        yield from read_csv(
            file,
            has_header=False,
            skip_rows=self.skip_lines,
            encoding=self.encoding,
            ignore_errors=self.errors,
            **self.fmtparams
        ).rows()
