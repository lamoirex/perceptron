import numpy as np
import pytest

from neuron.data import (
    load_iris_binary,
    make_two_clusters,
    standardize,
    train_test_split,
)
from neuron.model import Perceptron


def test_shapes_and_labels() -> None:
    X, y = make_two_clusters(n_samples=101, seed=0)
    assert X.shape == (101, 2)
    assert y.shape == (101,)
    assert set(np.unique(y)) == {0, 1}
    assert np.sum(y == 0) == 50
    assert np.sum(y == 1) == 51


def test_same_seed_gives_same_data() -> None:
    X1, y1 = make_two_clusters(n_samples=20, seed=42)
    X2, y2 = make_two_clusters(n_samples=20, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_clusters_are_around_their_centers() -> None:
    X, y = make_two_clusters(n_samples=1000, seed=1)
    assert np.allclose(X[y == 0].mean(axis=0), [-2.0, -2.0], atol=0.1)
    assert np.allclose(X[y == 1].mean(axis=0), [2.0, 2.0], atol=0.1)


@pytest.mark.parametrize(("n_samples", "spread"), [(1, 0.6), (10, 0.0)])
def test_invalid_arguments_raise(n_samples: int, spread: float) -> None:
    with pytest.raises(ValueError):
        make_two_clusters(n_samples, spread)


def test_iris_setosa_task_shape() -> None:
    X, y = load_iris_binary("setosa")
    assert X.shape == (150, 4)
    assert np.sum(y == 1) == 50


def test_iris_versicolor_task_shape() -> None:
    X, y = load_iris_binary("versicolor")
    assert X.shape == (100, 4)
    assert np.sum(y == 1) == 50


def test_split_keeps_all_samples_once() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3, seed=0)
    assert len(X_test) == 3
    assert len(X_train) == 7
    assert sorted(np.concatenate([y_train, y_test]).tolist()) == list(range(10))
    assert np.array_equal(X_train[:, 0] / 2, y_train)


def test_split_rejects_bad_ratio() -> None:
    X = np.zeros((10, 2))
    y = np.zeros(10, dtype=np.int_)
    with pytest.raises(ValueError):
        train_test_split(X, y, test_ratio=0.0)


def test_standardize_uses_train_statistics() -> None:
    X_train = np.array([[1.0, 10.0], [3.0, 10.0]])
    X_test = np.array([[5.0, 10.0]])
    train_scaled, test_scaled = standardize(X_train, X_test)
    assert np.allclose(train_scaled, [[-1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(test_scaled, [[3.0, 0.0]])


def test_perceptron_separates_setosa() -> None:
    X, y = load_iris_binary("setosa")
    model = Perceptron(n_features=4, epochs=20)
    model.train(X, y)
    assert model.score(X, y) == 1.0
