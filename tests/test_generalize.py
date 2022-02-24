import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.generalize import main
from satproc.postprocess.generalize import generalize

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_generalize_simplify(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_files = list(datadir.glob("*.gpkg"))

        generalize(
            input_files=input_files,
            output_dir=tmpdir,
            target_crs="epsg:3857",
        )

        output_path = Path(tmpdir) / input_files[0].name
        assert output_path.is_file()


def test_generalize_smooth(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_files = list(datadir.glob("*.gpkg"))

        generalize(
            input_files=input_files,
            output_dir=tmpdir,
            target_crs="epsg:3857",
            smooth="chaikin",
        )

        output_path = Path(tmpdir) / input_files[0].name
        assert output_path.is_file()


def test_cli_main(capsys):
    """CLI test"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html

    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert f"satproc {__version__}" in captured.out
