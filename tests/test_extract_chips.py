import tempfile
from pathlib import Path

import pytest

from satproc import __version__
from satproc.chips import extract_chips
from satproc.console.extract_chips import main

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"


def test_extract_chips_basic(shared_datadir):
    image_path = shared_datadir / "lux1.tif"

    with tempfile.TemporaryDirectory() as tmpdir:
        rasters = [str(image_path)]
        extract_chips(
            rasters,
            size=128,
            step_size=128,
            bands=[1, 2, 3, 4],
            output_dir=tmpdir,
        )
        output_images_dir = Path(tmpdir) / "images"
        assert output_images_dir.is_dir()
        assert len(list(output_images_dir.glob("*.tif"))) == 8


def test_extract_chips_with_masks(shared_datadir):
    image_path = shared_datadir / "lux1.tif"
    labels_path = shared_datadir / "lux1_gt.geojson"

    with tempfile.TemporaryDirectory() as tmpdir:
        rasters = [str(image_path)]
        extract_chips(
            rasters,
            labels=labels_path,
            size=128,
            step_size=128,
            bands=[1, 2, 3, 4],
            output_dir=tmpdir,
        )

        output_images_dir = Path(tmpdir) / "images"
        assert output_images_dir.is_dir()
        assert len(list(output_images_dir.glob("*.tif"))) == 5

        output_extent_masks_dir = Path(tmpdir) / "extent"
        assert output_extent_masks_dir.is_dir()
        assert len(list(output_extent_masks_dir.glob("*.tif"))) == 5


def test_extract_chips_with_masks_and_aoi(shared_datadir):
    image_path = shared_datadir / "lux1.tif"
    labels_path = shared_datadir / "lux1_gt.geojson"
    aoi_path = shared_datadir / "lux1_aoi.geojson"

    with tempfile.TemporaryDirectory() as tmpdir:
        rasters = [str(image_path)]
        extract_chips(
            rasters,
            aoi=str(aoi_path),
            labels=str(labels_path),
            size=128,
            step_size=128,
            bands=[1, 2, 3, 4],
            output_dir=tmpdir,
        )

        output_images_dir = Path(tmpdir) / "images"
        assert output_images_dir.is_dir()
        assert len(list(output_images_dir.glob("*.tif"))) == 4

        output_extent_masks_dir = Path(tmpdir) / "extent"
        assert output_extent_masks_dir.is_dir()
        assert len(list(output_extent_masks_dir.glob("*.tif"))) == 4


def test_cli_main(capsys):
    """CLI test"""
    # capsys is a pytest fixture that allows asserts agains stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html

    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    captured = capsys.readouterr()
    assert f"satproc {__version__}" in captured.out
