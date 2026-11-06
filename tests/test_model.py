import numpy as np
import pytest

from neuron.model import Perceptron


def test_weights_have_expected_length() -> None:
    model = Perceptron(n_features=3)
    assert model.weights.shape == (3,)
    assert np.all(model.weights == 0.0)
    assert model.bias == 0.0


def test_activate_threshold() -> None:
    assert Perceptron.activate(0.0) == 1
    assert Perceptron.activate(2.5) == 1
    assert Perceptron.activate(-1.0) == 0


@pytest.mark.parametrize(
    ("n_features", "learning_rate", "epochs"),
    [(0, 0.1, 10), (2, 0.0, 10), (2, 0.1, 0)],
)
def test_invalid_arguments_raise(
    n_features: int, learning_rate: float, epochs: int
) -> None:
    with pytest.raises(ValueError):
        Perceptron(n_features, learning_rate, epochs)
