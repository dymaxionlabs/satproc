import logging
import os

import numpy as np
import rasterio
from rasterio.windows import bounds
from shapely.geometry import box
from shapely.ops import transform
from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm

from satproc.utils import (rescale_intensity, sliding_windows,
                           write_chips_geojson)

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def extract_chips(raster,
                  contour_shapefile=None,
                  rescale_range=None,
                  bands=[1, 2, 3],
                  *,
                  size,
                  step_size,
                  output_dir):

    basename, _ = os.path.splitext(os.path.basename(raster))

    with rasterio.open(raster) as ds:
        _logger.info("Raster size: %s", (ds.width, ds.height))

        if ds.count < 3:
            raise RuntimeError(
                "Raster must have 3 bands corresponding to RGB channels")

        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(
            sliding_windows(win_size, win_step_size, ds.width, ds.height))
        chips = []

        for c, (window, (i, j)) in tqdm(list(enumerate(windows))):
            _logger.debug("%s %s", window, (i, j))
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            img = np.array([img[b - 1, :, :] for b in bands])

            if rescale_range:
                lower_cut, upper_cut = rescale_range
                img = rescale_intensity(img, lower_cut, upper_cut)

            img_path = os.path.join(
                output_dir, "{basename}_{x}_{y}.jpg".format(basename=basename,
                                                            x=i,
                                                            y=j))
            image_was_saved = write_image(img, img_path)
            if image_was_saved:
                chip_shape = box(*bounds(window, ds.transform))
                chip = (chip_shape, (c, i, j))
                chips.append(chip)

        geojson_path = os.path.join(output_dir, "{}.geojson".format(basename))
        write_chips_geojson(geojson_path,
                            chips,
                            crs=str(ds.crs),
                            basename=basename)


def write_image(img, path, percentiles=None):
    rgb = np.dstack(img[:3, :, :]).astype(np.uint8)
    if exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        imsave(path, rgb)
    return True
