from pathlib import Path

import numpy as np
import pytest

from neuron.data import make_two_clusters
from neuron.model import Perceptron
from neuron.plots import boundary_line, plot_decision_boundary, plot_errors


def make_model(weights: list[float], bias: float) -> Perceptron:
    model = Perceptron(n_features=2)
    model.weights = np.array(weights)
    model.bias = bias
    return model


def test_boundary_points_lie_on_line() -> None:
    model = make_model([1.0, 2.0], -4.0)
    line = boundary_line(model, (-5.0, 5.0), (-5.0, 5.0))
    assert line is not None
    xs, ys = line
    assert np.allclose(1.0 * xs + 2.0 * ys - 4.0, 0.0)


def test_vertical_boundary_when_second_weight_is_zero() -> None:
    model = make_model([2.0, 0.0], -2.0)
    line = boundary_line(model, (-5.0, 5.0), (-3.0, 3.0))
    assert line is not None
    xs, ys = line
    assert np.allclose(xs, [1.0, 1.0])
    assert np.allclose(ys, [-3.0, 3.0])


def test_no_boundary_for_zero_weights() -> None:
    model = Perceptron(n_features=2)
    assert boundary_line(model, (0.0, 1.0), (0.0, 1.0)) is None


def test_boundary_requires_two_features() -> None:
    with pytest.raises(ValueError):
        boundary_line(Perceptron(n_features=3), (0.0, 1.0), (0.0, 1.0))


def test_plots_are_saved(tmp_path: Path) -> None:
    X, y = make_two_clusters(n_samples=50, seed=0)
    model = Perceptron(n_features=2, epochs=10)
    errors = model.train(X, y)
    plot_decision_boundary(model, X, y, tmp_path / "boundary.png")
    plot_errors(errors, tmp_path / "errors.png")
    assert (tmp_path / "boundary.png").stat().st_size > 0
    assert (tmp_path / "errors.png").stat().st_size > 0
