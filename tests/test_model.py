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


def make_model_with_line() -> Perceptron:
    """Model with boundary x1 + x2 = 1 (class 1 above the line)."""
    model = Perceptron(n_features=2)
    model.weights = np.array([1.0, 1.0])
    model.bias = -1.0
    return model


@pytest.mark.parametrize(
    ("point", "expected"),
    [([2.0, 2.0], 1), ([0.0, 0.0], 0), ([-1.0, -3.0], 0), ([1.0, 0.0], 1)],
)
def test_predict_on_separable_points(point: list[float], expected: int) -> None:
    model = make_model_with_line()
    assert model.predict(np.array(point)) == expected


def test_predict_rejects_wrong_shape() -> None:
    model = make_model_with_line()
    with pytest.raises(ValueError):
        model.predict(np.array([1.0, 2.0, 3.0]))


AND_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
AND_Y = np.array([0, 0, 0, 1])


def test_score_counts_correct_fraction() -> None:
    model = make_model_with_line()
    # predictions are 0, 1, 1, 1; AND is 0, 0, 0, 1 -> 2 of 4 match
    assert model.score(AND_X, AND_Y) == 0.5


def test_train_learns_and_gate() -> None:
    model = Perceptron(n_features=2, learning_rate=1.0, epochs=20)
    model.train(AND_X, AND_Y)
    assert model.score(AND_X, AND_Y) == 1.0


def test_train_rejects_bad_labels() -> None:
    model = Perceptron(n_features=2)
    with pytest.raises(ValueError):
        model.train(AND_X, np.array([0, 1, 2, 1]))
