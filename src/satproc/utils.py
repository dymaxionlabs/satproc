import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
from itertools import zip_longest
from multiprocessing.pool import ThreadPool

import fiona
import numpy as np
import pyproj
import rasterio
from packaging import version
from pyproj.crs import CRS
from pyproj.enums import WktVersion
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.ops import transform
from skimage import exposure
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def sliding_windows(size, step_size, width, height, mode="exact"):
    """Slide a window of +size+ by moving it +step_size+ pixels

    Parameters
    ----------
    size : int
        window size, in pixels
    step_size : int
        step or *stride* size when sliding window, in pixels
    width : int
        image width
    height : int
        image height
    mode : str (default: 'exact')
        either one of 'exact', 'whole', 'whole_overlap'.
        - 'exact': clip windows at borders, if needed
        - 'whole': only whole windows
        - 'whole_overlap': only wohle windows, allow overlapping windows at borders.

    Yields
    ------
    Tuple[Window, Tuple[int, int]]
        a pair of Window and a pair of position (i, j)

    """
    w, h = size
    sw, sh = step_size

    whole = mode in ("whole", "whole_overlap")
    end_i = height - h if whole else height
    end_j = width - w if whole else width

    last_pos_i, last_pos_j = 0, 0
    for pos_i, i in enumerate(range(0, end_i, sh)):
        for pos_j, j in enumerate(range(0, end_j, sw)):
            real_w = w if whole else min(w, abs(width - j))
            real_h = h if whole else min(h, abs(height - i))
            yield Window(j, i, real_w, real_h), (pos_i, pos_j)
            last_pos_i, last_pos_j = pos_i, pos_j

    if mode == "whole_overlap" and (height % sh != 0 or width % sw != 0):
        for pos_i, i in enumerate(range(0, height - h, sh)):
            yield Window(width - w, i, w, h), (
                pos_i,
                last_pos_j + 1,
            )
        for pos_j, j in enumerate(range(0, width - w, sw)):
            yield Window(j, height - h, w, h), (
                last_pos_i + 1,
                pos_j,
            )
        yield Window(width - w, height - h, w, h), (last_pos_i + 1, last_pos_j + 1)


def rescale_intensity(image, rescale_mode, rescale_range):
    """
    Calculate percentiles from a range cut and
    rescale intensity of image to byte range

    Parameters
    ----------
    image : numpy.ndarray
        image array
    rescale_mode : str
        rescaling mode, either 'percentiles' or 'values'
    rescale_range : Tuple[number, number]
        input range for rescaling

    Returns
    -------
    numpy.ndarray
        rescaled image
    """

    if rescale_mode == "percentiles":
        in_range = np.percentile(image, rescale_range, axis=(1, 2)).T
    elif rescale_mode == "values":
        min_value, max_value = rescale_range
        if min_value is None:
            min_value = np.min(image)
        if max_value is None:
            max_value = np.max(image)
        in_range = np.array([(min_value, max_value) for _ in range(image.shape[0])])
    elif rescale_mode == "s2_rgb_extra":
        in_range = np.percentile(image, rescale_range, axis=(1, 2)).T
        # Override first 3 ranges for (0, 0.3) (Sentinel-2 L2A TCI range)
        in_range[0] = (0, 0.3)
        in_range[1] = (0, 0.3)
        in_range[2] = (0, 0.3)
    else:
        raise RuntimeError(f"unknown rescale_mode {rescale_mode}")

    return np.array(
        [
            exposure.rescale_intensity(
                image[i, :, :], in_range=tuple(in_range[i]), out_range=(1, 255)
            ).astype(np.uint8)
            for i in range(image.shape[0])
        ]
    )


def calculate_raster_percentiles(
    raster, lower_cut=2, upper_cut=98, sample_size=5000, block_size=2048
):
    """Estimate raster percentiles in blocks

    This method tiles raster image in blocks, samples block
    and calculates percentiles on all samples.

    Useful for large images that don't fit memory.

    Parameters
    ----------
    raster : str
        path to raster image
    lower_cut : float
        lower cut percentile (default: 2)
    upper_cut : float
        upper cut percentile (default: 98)
    sample_size : int
        how many samples to take for each block (default: 5000)
    block_size : int
        size of square block (default: 2048)

    Returns
    -------
    Tuple[float, float]
        raster percentiles

    """
    sample_size = 5000
    size = (2048, 2048)

    with rasterio.open(raster) as ds:
        windows = list(sliding_windows(size, size, ds.width, ds.height))
        window_sample_size = math.ceil(sample_size / len(windows))
        _logger.info(
            "Windows: %d, windows sample size: %d", len(windows), window_sample_size
        )
        totals_per_bands = [[] for _ in range(ds.count)]
        for window, _ in tqdm(windows):
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            window_sample = []
            for i in range(img.shape[0]):
                values = img[i].flatten()
                window_sample.append(
                    np.random.choice(values, size=window_sample_size, replace=False)
                )
            for i, win in enumerate(window_sample):
                totals_per_bands[i].append(win)

        for i, totals in enumerate(totals_per_bands):
            totals_per_bands[i] = np.array(totals).flatten()

        totals = np.array(totals_per_bands)
        _logger.info("Total shape: %s", totals.shape)

        res = tuple(
            tuple(p) for p in np.percentile(totals, (lower_cut, upper_cut), axis=1).T
        )
        _logger.info("Percentiles: %s", res)

        return res


def write_chips_geojson(output_path, chip_pairs, *, chip_type, crs, basename):
    """Write a GeoJSON containing chips polygons as features

    Parameters
    ----------
    output_path : str
        GeoJSON output path
    chip_pairs : Tuple[Shape, Tuple[int, int, int]]
        a pair with the chip polygon geometry, and a tuple of (feature id, x, y)
    chip_type : str
        chip file type extension (e.g. tif, jpg)
    crs : str
        CRS epsg code of chip polygon geometry
    basename : str
        basename of chip files

    Returns
    -------
    None

    """
    if not chip_pairs:
        _logger.warn("No chips to save")
        return

    _logger.info("Write chips geojson")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        d = {"type": "FeatureCollection", "features": []}
        if crs != "epsg:4326":
            code = crs.split(":")[1]
            d["crs"] = {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{code}"},
            }
        for i, (chip, (_fi, xi, yi)) in enumerate(chip_pairs):
            filename = f"{basename}_{xi}_{yi}.{chip_type}"
            feature = {
                "type": "Feature",
                "geometry": mapping(chip),
                "properties": {"id": i, "x": xi, "y": yi, "filename": filename},
            }
            d["features"].append(feature)
        f.write(json.dumps(d))


def get_raster_band_count(path):
    """Get raster band count

    Parameters
    ----------
    path : str
        path of the raster image

    Returns
    -------
    int
        band count

    """
    with rasterio.open(path) as src:
        return src.count


def reproject_shape(shp, from_crs, to_crs, project=None):
    """Reproject a shape from `from_crs` to `to_crs`

    Parameters
    ----------
    shp : Shape
        shape to reproject
    from_crs : str
        CRS epsg code of shape geometry
    to_crs : str
        CRS epsg code of reprojected shape geometry
    project : Optional[str]
        a Transformer instance to use for reprojection

    Returns
    -------
    Shape
        reprojected shape

    """
    if from_crs == to_crs:
        return shp
    if project is None:
        project = pyproj.Transformer.from_crs(
            from_crs, to_crs, always_xy=True
        ).transform
    return transform(project, shp)


def proj_crs_from_fiona_dataset(fio_ds):
    return CRS.from_wkt(fio_ds.crs_wkt)


def fiona_crs_from_proj_crs(proj_crs):
    if version.parse(fiona.__gdal_version__) < version.parse("3.0.0"):
        fio_crs = proj_crs.to_wkt(WktVersion.WKT1_GDAL)
    else:
        # GDAL 3+ can use WKT2
        fio_crs = proj_crs.to_wkt()
    return fio_crs


def run_command(cmd, quiet=True):
    """Run a shell command

    Parameters
    ----------
    cmd : str
        command to run
    quiet : bool (default: True)
        silent output (stdout and sterr)

    Returns
    -------
    None

    """
    stderr = subprocess.DEVNULL if quiet else None
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, shell=True, stderr=stderr, stdout=stdout)


def map_with_threads(items, worker, num_jobs=None, total=None, desc=None):
    """Map a worker function to an iterable of items, using a thread pool

    Parameters
    ----------
    items : iterable
        items to map
    worker : Function
        worker function to apply to each item
    num_jobs : int
        number of threads to use
    total : int (optional)
        total number of items (for the progress bar)
    desc : str (optional)
        description of the task (for the progress bar)

    Returns
    -------
    None

    """
    if not total:
        total = len(items)
    if not num_jobs:
        num_jobs = mp.cpu_count()
    with ThreadPool(num_jobs) as pool:
        with logging_redirect_tqdm():
            with tqdm(total=len(items), ascii=True, desc=desc) as pbar:
                for _ in enumerate(pool.imap_unordered(worker, items)):
                    pbar.update()
