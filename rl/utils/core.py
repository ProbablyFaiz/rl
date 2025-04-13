import collections
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import rl.utils.io
from rl.utils.logger import LOGGER

K = TypeVar("K")
T = TypeVar("T")

_MAP_MODE = rl.utils.io.getenv("RL_MAP_MODE", default=None)
if _MAP_MODE is not None:
    assert _MAP_MODE in (
        "none",
        "thread",
        "process",
    ), f"Invalid RL_MAP_MODE: {_MAP_MODE}"


def group_by(iterable: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    """Group an iterable by a key function into a dict of (key: list)."""
    groups = collections.defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)
    return dict(groups)


def dig(d: dict, *keys: str) -> Any:
    """Recursively get a value from a nested dictionary."""
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def safe_extract(pattern: re.Pattern, text: str, key: str) -> str | None:
    match = pattern.search(text)
    if match:
        return match.group(key)
    else:
        return None


def parallel_map(
    fn: Callable[..., T],
    *iterables: Iterable[Any],
    default_mode: str | None = None,
    **tqdm_kwargs: Any,
) -> list[T]:
    """Map a function over iterables with a tqdm progress bar."""
    import tqdm
    from tqdm.contrib.concurrent import process_map, thread_map

    if _MAP_MODE is None and default_mode is None:
        LOGGER.warning("RL_MAP_MODE is not set, using default mode: thread")
        mode = "thread"
    elif _MAP_MODE is not None and default_mode != _MAP_MODE:
        LOGGER.warning(
            "RL_MAP_MODE is set to %s, but default_mode is %s, using RL_MAP_MODE",
            _MAP_MODE,
            default_mode,
        )
        mode = _MAP_MODE
    else:
        mode = default_mode

    if mode == "thread":
        return thread_map(fn, *iterables, **tqdm_kwargs)
    elif mode == "process":
        return process_map(fn, *iterables, **tqdm_kwargs)
    else:
        for parallel_kwarg in ("max_workers", "chunksize"):
            if parallel_kwarg in tqdm_kwargs:
                tqdm_kwargs = tqdm_kwargs.copy()
                tqdm_kwargs.pop(parallel_kwarg)
        return list(tqdm.tqdm(map(fn, *iterables), **tqdm_kwargs))
