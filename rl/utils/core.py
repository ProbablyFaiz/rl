import collections
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import regex

import rl.utils.io

K = TypeVar("K")
T = TypeVar("T")

_MAP_MODE = rl.utils.io.getenv("RL_MAP_MODE", default=None)
if _MAP_MODE is not None:
    assert _MAP_MODE in ("thread", "process"), f"Invalid RL_MAP_MODE: {_MAP_MODE}"


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


def safe_extract(pattern: regex.Pattern, text: str, key: str) -> str | None:
    match = pattern.search(text)
    if match:
        return match.group(key)
    else:
        return None


def tqdm_map(
    fn: Callable[[T], Any],
    *iterables: Iterable[T],
    mode: str = _MAP_MODE,
    **tqdm_kwargs: Any,
) -> list[Any]:
    """Map a function over iterables with a tqdm progress bar."""
    import tqdm
    from tqdm.contrib.concurrent import process_map, thread_map

    if mode == "thread":
        return thread_map(fn, *iterables, **tqdm_kwargs)
    elif mode == "process":
        return process_map(fn, *iterables, **tqdm_kwargs)
    else:
        return list(tqdm.tqdm(map(fn, *iterables), **tqdm_kwargs))
