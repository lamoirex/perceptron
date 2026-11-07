"""Plots: decision boundary and training errors per epoch."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from neuron.model import FloatArray, IntArray, Perceptron

CLASS_COLORS = ("#2a78d6", "#eb6834")
CLASS_MARKERS = ("o", "s")
BOUNDARY_COLOR = "#52514e"
TEXT_COLOR = "#52514e"


def boundary_line(
    model: Perceptron,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[FloatArray, FloatArray] | None:
    """Compute two points of the decision boundary ``w1*x1 + w2*x2 + b = 0``.

    Args:
        model: Perceptron with exactly two features.
        x_range: Minimum and maximum of the first feature.
        y_range: Minimum and maximum of the second feature.

    Returns:
        Arrays ``(xs, ys)`` with two points each, or None if both weights
        are zero and the boundary does not exist.

    Raises:
        ValueError: If the model does not have exactly two features.
    """
    if model.n_features != 2:
        raise ValueError("decision boundary can be drawn only for 2 features")
    w1, w2 = float(model.weights[0]), float(model.weights[1])
    b = model.bias
    if w2 != 0:
        xs = np.array(x_range, dtype=np.float64)
        ys = -(w1 * xs + b) / w2
        return xs, ys
    if w1 != 0:
        x = -b / w1
        return np.array([x, x]), np.array(y_range, dtype=np.float64)
    return None


def _style_axes(fig: Figure, title: str, xlabel: str, ylabel: str) -> None:
    """Apply shared styling to the only axes of a figure.

    Args:
        fig: Figure with one axes.
        title: Axes title.
        xlabel: Label of the horizontal axis.
        ylabel: Label of the vertical axis.
    """
    ax = fig.axes[0]
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(color="#e3e2de", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_decision_boundary(
    model: Perceptron, X: FloatArray, y: IntArray, path: Path
) -> None:
    """Save a scatter plot of both classes with the learned boundary.

    Args:
        model: Trained perceptron with two features.
        X: Points of shape (n_samples, 2).
        y: Labels of shape (n_samples,).
        path: Output image file, e.g. ``boundary.png``.
    """
    fig = Figure(figsize=(6, 5), layout="constrained")
    ax = fig.add_subplot()
    for label in (0, 1):
        points = X[y == label]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            color=CLASS_COLORS[label],
            marker=CLASS_MARKERS[label],
            edgecolors="white",
            linewidths=0.6,
            label=f"class {label}",
        )

    x_range = (float(X[:, 0].min()) - 1, float(X[:, 0].max()) + 1)
    y_range = (float(X[:, 1].min()) - 1, float(X[:, 1].max()) + 1)
    line = boundary_line(model, x_range, y_range)
    if line is not None:
        ax.plot(*line, color=BOUNDARY_COLOR, linewidth=2, label="decision boundary")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.legend(frameon=False, loc="upper left")
    _style_axes(fig, "Perceptron decision boundary", "x1", "x2")
    fig.savefig(path, dpi=150)


def plot_errors(errors: list[int], path: Path) -> None:
    """Save a line chart of misclassified samples per epoch.

    Args:
        errors: Misclassified samples count for each epoch.
        path: Output image file, e.g. ``errors.png``.
    """
    fig = Figure(figsize=(6, 4), layout="constrained")
    ax = fig.add_subplot()
    epochs = range(1, len(errors) + 1)
    ax.plot(epochs, errors, color=CLASS_COLORS[0], linewidth=2, marker="o", ms=4)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(fig, "Training errors per epoch", "epoch", "misclassified samples")
    fig.savefig(path, dpi=150)
