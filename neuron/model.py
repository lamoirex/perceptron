"""Single-layer perceptron model."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    Attributes:
        n_features: Number of input features.
        learning_rate: Step size used when updating weights.
        epochs: Number of full passes over the training data.
        weights: Weight vector of shape (n_features,).
        bias: Scalar bias term that shifts the decision boundary.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Create a perceptron with zero-initialized weights and bias.

        Args:
            n_features: Number of input features, must be positive.
            learning_rate: Step size for weight updates, must be positive.
            epochs: Number of training epochs, must be positive.

        Raises:
            ValueError: If any argument is not positive.
        """
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: FloatArray = np.zeros(n_features, dtype=np.float64)
        self.bias: float = 0.0

    @staticmethod
    def activate(z: float) -> int:
        """Apply the step activation function.

        Args:
            z: Weighted sum of inputs plus bias.

        Returns:
            1 if ``z >= 0``, otherwise 0.
        """
        return 1 if z >= 0 else 0

    def net_input(self, x: FloatArray) -> float:
        """Compute the weighted sum of inputs plus bias.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            The value ``z = w · x + b``.

        Raises:
            ValueError: If ``x`` has the wrong shape.
        """
        if x.shape != (self.n_features,):
            raise ValueError(f"expected x of shape ({self.n_features},), got {x.shape}")
        return float(np.dot(self.weights, x)) + self.bias

