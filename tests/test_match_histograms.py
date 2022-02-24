import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.match_histograms import main
from satproc.histogram import match_histograms

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_match_histograms(datadir):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = datadir / "input.tif"
        ref_path = datadir / "ref.tif"
        output_path = Path(tmpdir) / "output.tif"

        match_histograms(
            src_path=input_path,
            reference_path=ref_path,
            dst_path=output_path,
        )

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
