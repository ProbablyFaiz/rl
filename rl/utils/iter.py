import collections
from typing import Callable, Iterable, TypeVar

K = TypeVar("K")
T = TypeVar("T")


def group_by(iterable: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    """Group an iterable by a key function into a dict of (key: list)."""
    groups = collections.defaultdict(list)
    for item in iterable:
        groups[key(item)].append(item)
    return dict(groups)


__all__ = ["group_by"]
