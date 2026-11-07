from pathlib import Path

import pytest

from neuron.cli import first_zero_error_epoch, main


def test_main_prints_accuracy(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--samples", "100", "--epochs", "20", "--seed", "3"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Accuracy: 100.00%" in output


@pytest.mark.parametrize(
    "args",
    [["--epochs", "0"], ["--samples", "abc"], ["--lr", "-0.5"]],
)
def test_invalid_arguments_exit_with_message(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_first_zero_error_epoch() -> None:
    assert first_zero_error_epoch([3, 1, 0, 0]) == 3
    assert first_zero_error_epoch([2, 2, 2]) is None


def test_main_saves_plots(tmp_path: Path) -> None:
    exit_code = main(["--samples", "40", "--seed", "1", "--plot-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "boundary.png").exists()
    assert (tmp_path / "errors.png").exists()
