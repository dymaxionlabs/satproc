import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
from functools import partial
from itertools import zip_longest
from multiprocessing.pool import ThreadPool

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.ops import transform
from skimage import exposure
from tqdm import tqdm

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def reproject_shape(shp, from_crs, to_crs):
    project = partial(pyproj.transform, pyproj.Proj(from_crs),
                      pyproj.Proj(to_crs))
    return transform(project, shp)


def sliding_windows(size, step_size, width, height, whole=False):
    """Slide a window of +size+ by moving it +step_size+ pixels"""
    w, h = size
    sw, sh = step_size
    end_i = height - h if whole else height
    end_j = width - w if whole else width
    for pos_i, i in enumerate(range(0, end_i, sh)):
        for pos_j, j in enumerate(range(0, end_j, sw)):
            real_w = w if whole else min(w, abs(width - j))
            real_h = h if whole else min(h, abs(height - i))
            yield Window(j, i, real_w, real_h), (pos_i, pos_j)


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


def write_chips_geojson(output_path, chip_pairs, *, type, crs, basename):
    if not chip_pairs:
        _logger.warn("No chips to save")
        return

    _logger.info("Write chips geojson")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        d = {"type": "FeatureCollection", "features": []}
        for i, (chip, (_fi, xi, yi)) in enumerate(chip_pairs):
            # Shapes will be stored in EPSG:4326 projection
            if crs != "epsg:4326":
                chip_wgs = reproject_shape(chip, crs, "epsg:4326")
            else:
                chip_wgs = chip
            filename = f"{basename}_{xi}_{yi}.{type}"
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


def get_raster_band_count(path):
    with rasterio.open(path) as src:
        return src.count


def run_command(cmd, quiet=True):
    stderr = subprocess.DEVNULL if quiet else None
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, shell=True, stderr=stderr, stdout=stdout)


def map_with_threads(items, worker, num_jobs=None, total=None):
    if not total:
        total = len(items)
    if not num_jobs:
        num_jobs = mp.cpu_count()
    with ThreadPool(num_jobs) as pool:
        with tqdm(total=len(items)) as pbar:
            for _ in enumerate(pool.imap_unordered(worker, items)):
                pbar.update()
