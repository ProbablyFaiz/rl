import glob
import json
import os
from pathlib import Path
from typing import Any, Iterable

import dotenv

DOTENV_LOADED = False


def ensure_dotenv_loaded():
    global DOTENV_LOADED
    if not DOTENV_LOADED:
        dotenv.load_dotenv()
        DOTENV_LOADED = True


def get_project_root() -> Path:
    """Get the root directory of the project."""
    if project_root := getenv("PROJECT_ROOT"):
        return Path(project_root)
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
    return get_project_root().joinpath("data", *args)


def get_model_path(*args) -> Path:
    """Get the path to a file nested in the models directory. If the MODELS_ROOT environment
    variable is set, use that as the root directory. Otherwise, use the models directory
    in the project root.

    Args:
        *args: The path components (strings or Path objects) to append to the models root.
    """
    if models_root := getenv("MODELS_ROOT"):
        return Path(models_root).joinpath(*args)
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


def getenv(name: str, default=None) -> str:
    """Get an environment variable. If the variable is not set, return the default value.

    Ensures that the .env file is loaded before attempting to get the environment variable.

    Args:
        name: The name of the environment variable.
        default: The default value to return if the environment variable is not set.
    """
    ensure_dotenv_loaded()
    return os.getenv(name, default)


def setenv(name: str, value: str) -> None:
    """Set an environment variable."""
    os.environ[name] = value


def read_jsonl(filename: str | Path) -> Iterable[Any]:
    with open(filename, "r") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(filename: str | Path, records: Iterable[Any], overwrite=False) -> None:
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.exists() and not overwrite:
        raise ValueError(f"{filename} already exists and overwrite is not set.")
    with open(filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_json(filename: str | Path) -> Any:
    with open(filename, "r") as f:
        return json.load(f)


def glob_to_files(glob_pattern: Path | str) -> list[Path]:
    """Convert a glob pattern to a list of files."""
    if isinstance(glob_pattern, str):
        glob_pattern = Path(glob_pattern)
    return sorted(map(Path, glob.glob(str(glob_pattern))))
