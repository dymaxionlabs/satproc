import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.console.make_masks import main
from satproc.masks import make_masks

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_make_masks(shared_datadir):
    # tests/data/lux1.tif --labels tests/data/lux1_gt.geojson
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = shared_datadir / "lux1.tif"
        labels_path = shared_datadir / "lux1_gt.geojson"

        rasters = [str(image_path)]
        make_masks(
            rasters=rasters, output_dir=tmpdir, labels=labels_path, classes=["A"]
        )

        output_extent_masks_dir = Path(tmpdir) / "extent"
        assert output_extent_masks_dir.is_dir()
        assert len(list(output_extent_masks_dir.glob("*.tif"))) == 1
        output_path = output_extent_masks_dir / "lux1.tif"
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
