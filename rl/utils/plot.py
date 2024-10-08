"""
This module provides a decorator-based interface for creating figures and tables.

>>> import pandas as pd
>>> from rl.utils import plot
>>> from rl.utils import io
>>> import matplotlib.pyplot as plt
>>> @plot.figure
... def my_plot():
...     plt.plot([1, 2, 3], [1, 2, 3])
...     plt.title("My Plot")
...     return plt.gcf()
>>> @plot.table
... def my_table():
...     return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
>>> plot.run()
>>> import os
>>> assert os.path.exists(io.get_figures_path() / "png" / "my_plot.png")
>>> assert os.path.exists(io.get_tables_path() / "csv" / "my_table.csv")
>>> plot.clean()
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from tqdm import tqdm

from rl.utils import io

PlotMethod = Callable[..., Figure]
TableMethod = Callable[..., DataFrame]

plot_methods: list[PlotMethod] = []
table_methods: list[TableMethod] = []


def figure(func: PlotMethod):
    plot_methods.append(func)
    return func


def table(func: TableMethod):
    table_methods.append(func)
    return func


def _make_dirs(plot_formats, table_formats):
    for p in plot_formats:
        (io.get_figures_path() / p).mkdir(exist_ok=True)
    for t in table_formats:
        (io.get_tables_path() / t).mkdir(exist_ok=True)


def clean(plot_formats=("png", "eps"), table_formats=("tex", "csv")):
    _make_dirs(plot_formats, table_formats)
    for pf in plot_formats:
        for f in Path(io.get_figures_path() / pf).glob(f"*.{pf}"):
            f.unlink()
    for tf in table_formats:
        for f in Path(io.get_tables_path() / tf).glob(f"*.{tf}"):
            f.unlink()


def run(plot_formats=("png", "eps"), table_formats=("tex", "csv")):
    pbar = tqdm(total=len(plot_methods) + len(table_methods))
    for pm in plot_methods[::-1]:
        figure = pm()
        pbar.set_description(f"Plotting {pm.__name__}")
        for pf in plot_formats:
            figure.savefig(
                io.get_figures_path() / pf / f"{pm.__name__}.{pf}", bbox_inches="tight"
            )
        plt.clf()
        plt.close()
        pbar.update(1)

    for tm in table_methods[::-1]:
        df = tm()
        pbar.set_description(f"Tabling {tm.__name__}")
        for tf in table_formats:
            df.to_csv(io.get_tables_path() / tf / f"{tm.__name__}.{tf}")
        pbar.update(1)


if __name__ == "__main__":
    import doctest

    doctest.testmod(globs={"plot_methods": {}, "table_methods": {}})
