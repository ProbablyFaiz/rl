import csv
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dotenv
import requests
import tqdm

DOTENV_LOADED = False


def ensure_dotenv_loaded():
    global DOTENV_LOADED
    if not DOTENV_LOADED:
        dotenv.load_dotenv()
        DOTENV_LOADED = True


def get_project_root() -> Path:
    """Get the path to the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_path(*args) -> Path:
    """Get the path to a file nested in the data directory. If the DATA_ROOT environment
    variable is set, use that as the root directory. Otherwise, use the data directory
    in the project root.

    Args:
        *args: The path components (strings or Path objects) to append to the data root.
    """
    if data_root := getenv("DATA_ROOT"):
        return Path(data_root).joinpath(*args)
    raise ValueError("DATA_ROOT environment variable is not set.")


def get_cache_dir(*args) -> Path:
    return get_data_path("cache", *args)


def get_model_path(*args) -> Path:
    return get_data_path("models", *args)


def get_figures_path(*args) -> Path:
    """Get the path to a file nested in the figures directory. If the FIGURES_ROOT environment
    variable is set, use that as the root directory. Otherwise, use the figures directory
    in the project root.

    Args:
        *args: The path components (strings or Path objects) to append to the figures root.
    """
    if figures_root := getenv("FIGURES_ROOT"):
        return Path(figures_root).joinpath(*args)
    return get_project_root().joinpath("figures", *args)


def get_tables_path(*args) -> Path:
    """Get the path to a file nested in the figures directory. If the FIGURES_ROOT environment
    variable is set, use that as the root directory. Otherwise, use the figures directory
    in the project root.

    Args:
        *args: The path components (strings or Path objects) to append to the figures root.
    """
    if tables_root := getenv("TABLES_ROOT"):
        return Path(tables_root).joinpath(*args)
    return get_project_root().joinpath("tables", *args)


def getenv(name: str, default=None) -> str:
    """Get an environment variable. If the variable is not set, return the default value.

    Ensures that the .env file is loaded before attempting to get the environment variable.

    Args:
        name: The name of the environment variable.
        default: The default value to return if the environment variable is not set.
    """
    ensure_dotenv_loaded()
    return os.getenv(name, default)


def read_jsonl(filename: str | Path) -> Iterable[Any]:
    filename = Path(filename)
    with filename.open() as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(filename: str | Path, records: Iterable[Any], overwrite=False) -> None:
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise ValueError(f"{filename} already exists and overwrite is not set.")
    with filename.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_jsonl_spark(filename: str | Path, df, overwrite=False) -> None:
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise ValueError(f"{filename} already exists and overwrite is not set.")
    output_path_dir_name = filename.parent / f"{filename.stem}_dir"
    df.coalesce(1).write.json(str(output_path_dir_name), lineSep="\n", mode="overwrite")
    output_path_dir = list(output_path_dir_name.glob("*.json"))[0]
    shutil.move(output_path_dir, filename)
    shutil.rmtree(output_path_dir_name)


def write_parquet_spark(filename: str | Path, df, overwrite=False) -> None:
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise ValueError(f"{filename} already exists and overwrite is not set.")

    output_path_dir_name = filename.parent / f"{filename.stem}_dir"
    df.coalesce(1).write.parquet(str(output_path_dir_name), mode="overwrite")
    parquet_file = list(output_path_dir_name.glob("*.parquet"))[0]
    shutil.move(parquet_file, filename)
    shutil.rmtree(output_path_dir_name)


def read_csv(filename: str | Path) -> Iterable[dict[str, Any]]:
    filename = Path(filename)
    with filename.open() as f:
        reader = csv.DictReader(f)
        yield from reader


def write_csv(
    filename: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    field_names: list[str] | None = None,
    overwrite=False,
) -> None:
    filename = Path(filename)
    if filename.exists() and not overwrite:
        raise ValueError(f"{filename} already exists and overwrite is not set.")
    kwargs = (
        {
            "fieldnames": field_names,
        }
        if field_names
        else {}
    )
    with filename.open("w") as f:
        writer = csv.DictWriter(f, **kwargs)  # type: ignore
        writer.writeheader()
        writer.writerows(records)


def download(url: str, dest: str | Path) -> None:
    dest = Path(dest)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with (
        dest.open("wb") as f,
        tqdm.tqdm(
            desc=str(dest),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            pbar.update(len(data))
