import math

import pytest

from entropy_agent_eval.metrics import (
    entropy_reduction,
    exploration_efficiency,
    normalized_entropy,
    rolling_entropy_curve,
    shannon_entropy,
)


def test_shannon_entropy_for_symbols():
    assert shannon_entropy(["search", "search", "answer", "answer"]) == pytest.approx(1.0)


def test_entropy_for_probability_vector():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_normalized_entropy():
    assert normalized_entropy(["a", "b", "c"]) == pytest.approx(1.0)
    assert normalized_entropy(["a", "a", "a"]) == pytest.approx(0.0)


def test_information_gain_positive_when_uncertainty_reduces():
    gain = entropy_reduction([0.4, 0.3, 0.2, 0.1], [0.9, 0.05, 0.03, 0.02])
    assert gain > 0


def test_rolling_entropy_curve_validates_window():
    with pytest.raises(ValueError):
        rolling_entropy_curve(["a"], 0)


def test_exploration_efficiency_rejects_invalid_success_rate():
    with pytest.raises(ValueError):
        exploration_efficiency(math.nan, 1.0)
