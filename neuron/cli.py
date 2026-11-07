"""Command-line interface for training the perceptron."""

import argparse


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
