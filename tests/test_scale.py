import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.scale import main
from satproc.scale import scale

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_scale(shared_datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = shared_datadir / "lux1.tif"
        output_path = Path(tmpdir) / "output.tif"

        scale(input_img=input_path, output_img=output_path)

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
