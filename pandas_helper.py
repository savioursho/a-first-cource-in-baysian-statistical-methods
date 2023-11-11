import pandas as pd
import numpy as np


class Column:
    def __init__(self, col_name):
        self.col_name = col_name
        self.lazy_series = None
        self.select_column()
        self.df = None

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        self.df = df
        return next(self.lazy_series)

    def _select_column(self):
        yield self.df[self.col_name].copy()

    def select_column(self):
        self.lazy_series = self._select_column()

    def _add(self, lazy_series, *args, **kwargs):
        yield next(lazy_series).add(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.lazy_series = self._add(self.lazy_series, *args, **kwargs)
        return self

    def _sub(self, lazy_series, *args, **kwargs):
        yield next(lazy_series).sub(*args, **kwargs)

    def sub(self, *args, **kwargs):
        self.lazy_series = self._sub(self.lazy_series, *args, **kwargs)
        return self


def col(col_name: str) -> pd.Series:
    return Column(col_name)


if __name__ == "__main__":
    # example
    df = pd.DataFrame(np.arange(100).reshape(25, 4), columns=list("ABCD"))
    df.assign(E=col("A").add(1))
