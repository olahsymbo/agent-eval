from __future__ import annotations

from typing import Hashable, Iterable, List

from entropy_agent_eval.metrics.core import shannon_entropy


def entropy_curve(symbols: Iterable[Hashable]) -> List[float]:
    """Cumulative entropy after each observed symbol."""

    seen: list[Hashable] = []
    curve: list[float] = []
    for symbol in symbols:
        seen.append(symbol)
        curve.append(shannon_entropy(seen))
    return curve


def rolling_entropy_curve(symbols: Iterable[Hashable], window_size: int) -> List[float]:
    """Rolling-window entropy for local behavior changes."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    values = list(symbols)
    return [
        shannon_entropy(values[max(0, index - window_size + 1) : index + 1])
        for index in range(len(values))
    ]
