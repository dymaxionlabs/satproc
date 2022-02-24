import tempfile
from pathlib import Path

import fiona
import pytest

from satproc import __version__
from satproc.console.polygonize import main
from satproc.postprocess.polygonize import polygonize

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_polygonize(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = datadir / "chips"
        output_path = Path(tmpdir) / "output.gpkg"

        polygonize(
            input_dir=str(input_dir),
            output=str(output_path),
        )

        assert output_path.is_file()
        with fiona.open(output_path) as src:
            features = list(src)
            assert len(features) == 2


def test_polygonize_with_threshold(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = datadir / "chips"
        output_path = Path(tmpdir) / "output.gpkg"

        polygonize(
            input_dir=str(input_dir),
            output=str(output_path),
            threshold=0.02,
        )

        assert output_path.is_file()
        with fiona.open(output_path) as src:
            features = list(src)
            assert len(features) == 2


def test_cli_main(capsys):
    """CLI test"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html

    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert f"satproc {__version__}" in captured.out
