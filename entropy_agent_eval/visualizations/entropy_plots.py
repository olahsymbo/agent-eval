from __future__ import annotations

from pathlib import Path
from typing import Iterable


def plot_entropy_curve(curve: Iterable[float], path: str | Path | None = None):
    """Plot an entropy curve with matplotlib.

    Returns the matplotlib axes. If ``path`` is provided, the figure is saved.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install entropy-agent-eval[plots] to use plotting helpers") from exc

    values = list(curve)
    _, ax = plt.subplots()
    ax.plot(range(1, len(values) + 1), values, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Agent Entropy Curve")
    ax.grid(True, alpha=0.25)
    if path is not None:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax
