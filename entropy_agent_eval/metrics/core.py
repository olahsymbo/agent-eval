from __future__ import annotations

import math
from collections import Counter
from typing import Hashable, Iterable, Sequence


def safe_probability_vector(values: Sequence[float]) -> list[float]:
    """Return a normalized probability vector, rejecting invalid inputs."""

    if not values:
        return []
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("probabilities/counts must be finite non-negative numbers")
    total = float(sum(values))
    if total <= 0:
        return []
    return [float(value) / total for value in values if value > 0]


def shannon_entropy(items_or_probabilities: Iterable[Hashable] | Sequence[float], base: float = 2.0) -> float:
    """Compute Shannon entropy for symbols or an explicit probability/count vector."""

    values = list(items_or_probabilities)
    if not values:
        return 0.0

    if all(isinstance(value, (int, float)) for value in values):
        probs = safe_probability_vector([float(value) for value in values])
    else:
        counts = Counter(values)
        probs = safe_probability_vector(list(counts.values()))

    if not probs:
        return 0.0
    log_base = math.log(base)
    value = -sum(prob * (math.log(prob) / log_base) for prob in probs)
    return 0.0 if abs(value) < 1e-12 else value


def normalized_entropy(items_or_probabilities: Iterable[Hashable] | Sequence[float], base: float = 2.0) -> float:
    """Entropy divided by the maximum possible entropy for the observed support size."""

    values = list(items_or_probabilities)
    if not values:
        return 0.0
    support = len([value for value in values if value > 0]) if _numeric(values) else len(set(values))
    if support <= 1:
        return 0.0
    return shannon_entropy(values, base=base) / (math.log(support) / math.log(base))


def entropy_reduction(before: Sequence[float], after: Sequence[float], base: float = 2.0) -> float:
    """Information gain as H(before) - H(after)."""

    return shannon_entropy(before, base=base) - shannon_entropy(after, base=base)


def exploration_efficiency(success_rate: float, entropy_value: float, epsilon: float = 1e-9) -> float:
    """Success per bit of entropy.

    The epsilon avoids division by zero while preserving the intended ranking.
    """

    if not math.isfinite(success_rate) or success_rate < 0 or success_rate > 1:
        raise ValueError("success_rate must be in [0, 1]")
    if not math.isfinite(entropy_value) or entropy_value < 0:
        raise ValueError("entropy_value must be a finite non-negative number")
    return success_rate / max(entropy_value, epsilon)


def _numeric(values: Sequence[object]) -> bool:
    return all(isinstance(value, (int, float)) for value in values)
