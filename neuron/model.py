"""Single-layer perceptron model."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    Attributes:
        n_features: Number of input features.
        learning_rate: Step size used when updating weights.
        epochs: Number of full passes over the training data.
        weights: Weight vector of shape (n_features,).
        bias: Scalar bias term that shifts the decision boundary.
        errors_per_epoch: Number of misclassified samples in each epoch.
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
        self.errors_per_epoch: list[int] = []

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

    def predict(self, x: FloatArray) -> int:
        """Predict the class of a single sample.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            Predicted class label, 0 or 1.
        """
        return self.activate(self.net_input(x))

    def _check_dataset(self, X: FloatArray, y: IntArray) -> None:
        """Validate that ``X`` and ``y`` form a correct dataset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,), each 0 or 1.

        Raises:
            ValueError: If shapes do not match or labels are not 0/1.
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"expected X of shape (n_samples, {self.n_features}), got {X.shape}"
            )
        if y.shape != (X.shape[0],):
            raise ValueError(f"expected y of shape ({X.shape[0]},), got {y.shape}")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("labels in y must be 0 or 1")

    def train(self, X: FloatArray, y: IntArray) -> list[int]:
        """Fit weights and bias with the perceptron learning rule.

        For every sample the update is ``w += lr * (target - pred) * x``
        and ``b += lr * (target - pred)``.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,), each 0 or 1.

        Returns:
            Number of misclassified samples in each epoch.
        """
        self._check_dataset(X, y)
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y, strict=True):
                error = int(target) - self.predict(xi)
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_per_epoch.append(errors)
        return self.errors_per_epoch

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            Fraction of correctly classified samples, from 0.0 to 1.0.
        """
        self._check_dataset(X, y)
        predictions = np.array([self.predict(xi) for xi in X])
        return float(np.mean(predictions == y))
