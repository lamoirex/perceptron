"""Command-line interface for training the perceptron."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from neuron.data import (
    IrisTask,
    load_iris_binary,
    make_two_clusters,
    standardize,
    train_test_split,
)
from neuron.model import Perceptron
from neuron.plots import plot_decision_boundary, plot_errors


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
        description="Train a perceptron on synthetic 2D points or the iris dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=["clusters", "iris-setosa", "iris-versicolor"],
        default="clusters",
        help="clusters: synthetic 2D points; iris-*: real iris flowers",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=200,
        help="number of points (clusters only)",
    )
    parser.add_argument("--lr", type=positive_float, default=0.1, help="learning rate")
    parser.add_argument(
        "--epochs", type=positive_int, default=50, help="number of epochs"
    )
    parser.add_argument(
        "--spread",
        type=positive_float,
        default=0.6,
        help="cloud standard deviation (clusters only)",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None, help="folder to save PNG plots into"
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


def print_training_report(model: Perceptron, errors: list[int]) -> None:
    """Print hyperparameters, convergence info and learned parameters.

    Args:
        model: Trained perceptron.
        errors: Misclassified samples count for each epoch.
    """
    print(f"Learning rate: {model.learning_rate}, epochs: {model.epochs}")
    converged = first_zero_error_epoch(errors)
    if converged is None:
        print(f"Not converged: {errors[-1]} errors in the last epoch")
    else:
        print(f"Converged at epoch {converged}")
    print(f"Weights: {model.weights.round(3).tolist()}, bias: {model.bias:.3f}")


def run_clusters(args: argparse.Namespace) -> None:
    """Train on synthetic 2D clusters, print the report and save plots.

    Args:
        args: Parsed command-line arguments.
    """
    X, y = make_two_clusters(n_samples=args.samples, spread=args.spread, seed=args.seed)
    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X, y)

    print(f"Dataset: two clusters, {args.samples} samples")
    print_training_report(model, errors)
    print(f"Accuracy: {model.score(X, y):.2%}")

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_decision_boundary(model, X, y, args.plot_dir / "boundary.png")
        plot_errors(errors, args.plot_dir / "errors.png")
        print(f"Plots saved to {args.plot_dir}/")


def run_iris(args: argparse.Namespace) -> None:
    """Train on iris with a train/test split, print the report and save plots.

    Args:
        args: Parsed command-line arguments.
    """
    task: IrisTask = "setosa" if args.dataset == "iris-setosa" else "versicolor"
    X, y = load_iris_binary(task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=args.seed)
    X_train, X_test = standardize(X_train, X_test)
    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X_train, y_train)

    print(f"Dataset: iris, task {task}, {len(X_train)} train / {len(X_test)} test")
    print_training_report(model, errors)
    print(f"Train accuracy: {model.score(X_train, y_train):.2%}")
    print(f"Test accuracy: {model.score(X_test, y_test):.2%}")

    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_errors(errors, args.plot_dir / "errors.png")
        print(f"Errors plot saved to {args.plot_dir}/ (boundary needs 2 features)")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line program.

    Args:
        argv: Arguments without the program name; None means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on error.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.dataset == "clusters":
            run_clusters(args)
        else:
            run_iris(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
