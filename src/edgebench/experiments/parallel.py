"""Run independent jobs on worker processes; each worker loads the corpus once."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def _init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from edgebench.bench import corpus

    corpus()


def run(fn: Callable[..., T], jobs: Iterable[tuple[Any, ...]], workers: int) -> list[T]:
    jobs = list(jobs)
    if workers <= 1:
        return [fn(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as ex:
        futures = [ex.submit(fn, *j) for j in jobs]
        return [f.result() for f in futures]
