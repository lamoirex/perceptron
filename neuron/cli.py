"""Command-line interface for training the perceptron."""

import argparse
import sys
from collections.abc import Sequence

from neuron.data import make_two_clusters
from neuron.model import Perceptron


def positive_int(text: str) -> int:
    """Parse a strictly positive integer for argparse.

    Args:
        text: Raw command-line value.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def positive_float(text: str) -> float:
    """Parse a strictly positive float for argparse.

    Args:
        text: Raw command-line value.

    Returns:
        The parsed float.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive number.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="perceptron.py",
        description="Train a perceptron on two synthetic 2D point clouds.",
    )
    parser.add_argument(
        "--samples", type=positive_int, default=200, help="number of points"
    )
    parser.add_argument("--lr", type=positive_float, default=0.1, help="learning rate")
    parser.add_argument(
        "--epochs", type=positive_int, default=50, help="number of epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )
    return parser


def first_zero_error_epoch(errors: list[int]) -> int | None:
    """Find the first epoch (1-based) without misclassifications.

    Args:
        errors: Misclassified samples count for each epoch.

    Returns:
        Epoch number starting from 1, or None if every epoch had errors.
    """
    for epoch, count in enumerate(errors, start=1):
        if count == 0:
            return epoch
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line program.

    Args:
        argv: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on error.
    """
    args = build_parser().parse_args(argv)
    try:
        X, y = make_two_clusters(n_samples=args.samples, seed=args.seed)
        model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
        errors = model.train(X, y)
        accuracy = model.score(X, y)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    converged = first_zero_error_epoch(errors)
    print(f"Samples: {args.samples}, learning rate: {args.lr}, epochs: {args.epochs}")
    if converged is None:
        print(f"Not converged: {errors[-1]} errors in the last epoch")
    else:
        print(f"Converged at epoch {converged}")
    print(f"Weights: {model.weights.round(3).tolist()}, bias: {model.bias:.3f}")
    print(f"Accuracy: {accuracy:.2%}")
    return 0
