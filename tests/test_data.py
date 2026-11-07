import numpy as np
import pytest

from neuron.data import make_two_clusters


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
