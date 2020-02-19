import json
import logging
import math
import os
from functools import partial

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window
from shapely.geometry import mapping
from skimage import exposure
from tqdm import tqdm

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def sliding_windows(size, step_size, width, height):
    """Slide a window of +size+ by moving it +step_size+ pixels"""
    w, h = size
    sw, sh = step_size
    for pos_i, i in enumerate(range(0, height - h + 1, sh)):
        for pos_j, j in enumerate(range(0, width - w + 1, sw)):
            yield Window(j, i, w, h), (pos_i, pos_j)


def rescale_intensity(image, rescale_mode, rescale_range):
    """Calculate percentiles from a range cut and rescale intensity of image"""

    if rescale_mode == 'percentiles':
        in_range = tuple(np.percentile(image, rescale_range))
    elif rescale_mode == 'values':
        min_value, max_value = rescale_range
        if not min_value:
            min_value = np.min(image)
        if not max_value:
            max_value = np.max(image)
        in_range = (min_value, max_value)

    return exposure.rescale_intensity(image,
                                      in_range=in_range,
                                      out_range=(0, 255)).astype(np.uint8)


def calculate_raster_percentiles(raster, lower_cut=2, upper_cut=98):
    sample_size = 5000
    size = (2048, 2048)

    with rasterio.open(raster) as ds:
        windows = list(sliding_windows(size, size, ds.width, ds.height))
        window_sample_size = math.ceil(sample_size / len(windows))
        _logger.info("Windows: %d, windows sample size: %d", len(windows),
                     window_sample_size)
        totals_per_bands = [[] for _ in range(ds.count)]
        for window, _ in tqdm(windows):
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            window_sample = []
            for i in range(img.shape[0]):
                values = img[i].flatten()
                window_sample.append(
                    np.random.choice(values,
                                     size=window_sample_size,
                                     replace=False))
            for i, win in enumerate(window_sample):
                totals_per_bands[i].append(win)

        for i, totals in enumerate(totals_per_bands):
            totals_per_bands[i] = np.array(totals).flatten()

        totals = np.array(totals_per_bands)
        _logger.info("Total shape: %s", totals.shape)

        res = tuple(
            tuple(p)
            for p in np.percentile(totals, (lower_cut, upper_cut), axis=1).T)
        _logger.info("Percentiles: %s", res)

        return res


def write_chips_geojson(output_path, chip_pairs, *, crs, basename):
    if not chip_pairs:
        _logger.warn("No chips to save")
        return

    _logger.info("Write chips geojson")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        d = {"type": "FeatureCollection", "features": []}
        for i, (chip, (fi, xi, yi)) in enumerate(chip_pairs):
            # Shapes will be stored in EPSG:4326 projection
            if crs != "EPSG:4326":
                project = partial(pyproj.transform, pyproj.Proj(crs),
                                  pyproj.Proj("EPSG:4326"))
                chip_wgs = transform(project, chip)
            else:
                chip_wgs = chip
            filename = "{basename}_{x}_{y}.jpg".format(basename=basename,
                                                       x=xi,
                                                       y=yi)
            feature = {
                "type": "Feature",
                "geometry": mapping(chip_wgs),
                "properties": {
                    "id": i,
                    "x": xi,
                    "y": yi,
                    "filename": filename
                },
            }
            d["features"].append(feature)
        f.write(json.dumps(d))
