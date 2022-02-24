import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.smooth_stitch import main
from satproc.postprocess.smooth import smooth_stitch

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_smooth(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = datadir / "distance"
        smooth_stitch(
            input_dir=input_dir,
            output_dir=tmpdir,
        )

        output_dir = Path(tmpdir)
        assert output_dir.is_dir()
        assert len(list(output_dir.glob("*.tif"))) == 8


def test_cli_main(capsys):
    """CLI test"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html

    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert f"satproc {__version__}" in captured.out
