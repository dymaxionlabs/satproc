import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.filter import main
from satproc.filter import filter_by_max_prob

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_filter_valid_threshold(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = datadir / "chips"

        filter_by_max_prob(
            input_dir=input_dir,
            output_dir=str(tmpdir),
            threshold=0.05,
        )

        output_images_dir = Path(tmpdir)
        assert output_images_dir.is_dir()
        assert len(list(output_images_dir.glob("*.tif"))) == 4


def test_filter_threshold_too_high(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = datadir / "chips"

        filter_by_max_prob(
            input_dir=input_dir,
            output_dir=str(tmpdir),
            threshold=0.5,
        )

        output_images_dir = Path(tmpdir)
        assert output_images_dir.is_dir()
        assert len(list(output_images_dir.glob("*.tif"))) == 0


def test_cli_main(capsys):
    """CLI test"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html

    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert f"satproc {__version__}" in captured.out
