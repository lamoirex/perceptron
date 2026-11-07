"""Synthetic and real datasets for the perceptron."""

import numpy as np

from neuron.model import FloatArray, IntArray


def make_two_clusters(
    n_samples: int,
    spread: float = 0.6,
    seed: int | None = None,
) -> tuple[FloatArray, IntArray]:
    """Generate two 2D point clouds that are linearly separable.

    Class 0 is scattered around (-2, -2), class 1 around (2, 2).
    Points of each class follow a normal (Gaussian) distribution.

    Args:
        n_samples: Total number of points, must be at least 2.
        spread: Standard deviation of each cloud, must be positive.
        seed: Random seed for reproducible results; None means random.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape (n_samples, 2) and
        ``y`` has shape (n_samples,) with labels 0 and 1.

    Raises:
        ValueError: If ``n_samples`` or ``spread`` is invalid.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if spread <= 0:
        raise ValueError(f"spread must be > 0, got {spread}")

    rng = np.random.default_rng(seed)
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    points0 = rng.normal(loc=(-2.0, -2.0), scale=spread, size=(n_class0, 2))
    points1 = rng.normal(loc=(2.0, 2.0), scale=spread, size=(n_class1, 2))

    X = np.vstack([points0, points1])
    y = np.concatenate(
        [np.zeros(n_class0, dtype=np.int_), np.ones(n_class1, dtype=np.int_)]
    )

    order = rng.permutation(n_samples)
    return X[order], y[order]
