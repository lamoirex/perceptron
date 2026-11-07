"""Synthetic and real datasets for the perceptron."""

from typing import Literal

import numpy as np
from sklearn.datasets import load_iris

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


IrisTask = Literal["setosa", "versicolor"]


def load_iris_binary(task: IrisTask) -> tuple[FloatArray, IntArray]:
    """Load the iris dataset as a binary classification problem.

    The iris dataset has 150 flowers of 3 species and 4 features
    (sepal length, sepal width, petal length, petal width, in cm).

    Args:
        task: ``"setosa"`` means setosa (1) vs the two other species (0),
            150 samples, linearly separable. ``"versicolor"`` means
            versicolor (0) vs virginica (1), 100 samples, not separable.

    Returns:
        Tuple ``(X, y)`` with X of shape (n_samples, 4) and 0/1 labels.
    """
    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    species = np.asarray(iris.target, dtype=np.int_)
    if task == "setosa":
        return X, (species == 0).astype(np.int_)
    mask = species != 0
    return X[mask], (species[mask] == 2).astype(np.int_)


def train_test_split(
    X: FloatArray, y: IntArray, test_ratio: float = 0.3, seed: int | None = None
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Randomly split a dataset into training and test parts.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Labels of shape (n_samples,).
        test_ratio: Fraction of samples for the test part, from 0 to 1.
        seed: Random seed for reproducible results; None means random.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If ``test_ratio`` leaves one of the parts empty.
    """
    n_test = round(len(X) * test_ratio)
    if not 0 < n_test < len(X):
        raise ValueError(f"test_ratio {test_ratio} leaves an empty train or test set")
    order = np.random.default_rng(seed).permutation(len(X))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(
    X_train: FloatArray, X_test: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Scale features to zero mean and unit variance.

    Mean and standard deviation are computed on the training part only,
    so no information from the test part leaks into training.

    Args:
        X_train: Training features of shape (n_train, n_features).
        X_test: Test features of shape (n_test, n_features).

    Returns:
        Tuple ``(X_train_scaled, X_test_scaled)``.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std
