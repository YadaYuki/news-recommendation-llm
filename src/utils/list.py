from typing import TypeVar

T = TypeVar("T")


def uniq(ls: list[T]) -> list[T]:
    return list(set(ls))
