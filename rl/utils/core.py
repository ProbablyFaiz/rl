import collections
from typing import Any, Callable, Iterable, TypeVar

import regex
from tqdm.contrib.concurrent import process_map, thread_map

K = TypeVar("K")
T = TypeVar("T")


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
    mode: str = None,
    **tqdm_kwargs: Any,
) -> list[Any]:
    """Map a function over iterables with a tqdm progress bar."""
    if mode == "thread":
        return thread_map(fn, *iterables, **tqdm_kwargs)
    elif mode == "process":
        return process_map(fn, *iterables, **tqdm_kwargs)
    else:
        return list(map(fn, *iterables))
