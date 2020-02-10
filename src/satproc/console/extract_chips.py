# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = satproc.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import logging

import rasterio
import fiona
import logging
import os
import math
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window
from skimage.io import imsave
from skimage import exposure

from satproc import __version__

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


def calculate_percentiles(raster, lower_cut=2, upper_cut=98):
    sample_size = 5000
    size = (2048, 2048)

    with rasterio.open(raster) as ds:
        windows = list(sliding_windows(size, size, ds.width, ds.height))
        window_sample_size = math.ceil(sample_size / len(windows))
        _logger.info("Windows: %d, windows sample size: %d", len(windows), window_sample_size)
        totals_per_bands = [[] for _ in range(ds.count)]
        for window, _ in tqdm(windows):
            img = ds.read(window=window)
            window_sample = []
            for i in range(img.shape[0]):
                values = img[i].flatten()
                window_sample.append(np.random.choice(values,
                        size=window_sample_size,
                        replace=False))
            for i, win in enumerate(window_sample):
                totals_per_bands[i].append(win)

        for i, totals in enumerate(totals_per_bands):
            totals_per_bands[i] = np.array(totals).flatten()

        totals = np.array(totals_per_bands)
        _logger.info("Total shape: %s", totals.shape)

        res = tuple(tuple(p) for p in np.percentile(totals, (lower_cut, upper_cut), axis=1).T)
        _logger.info("Percentiles: %s", res)

        return res


def extract_chips(raster, contour_shapefile=None, percentiles=None, *, size,
        step_size, output_dir):
    with rasterio.open(raster) as ds:
        _logger.info('Raster size: %s', (ds.width, ds.height))

        if ds.count < 3:
            raise RuntimeError("Raster must have 3 bands corresponding to RGB channels")

        if ds.count > 3:
            _logger.warn("WARNING: Raster has {} bands. " \
                  "Going to assume first 3 bands are RGB...".format(ds.count))

        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(sliding_windows(win_size, win_step_size, ds.width, ds.height))
        saved_windows = []

        for window, (i, j) in tqdm(windows):
            _logger.debug("%s %s", window, (i, j))
            img = ds.read(window=window)
            img = img[:3, :, :]

            if percentiles:
                new_img = []
                for k, perc in enumerate(percentiles):
                    band = img[k, :, :]
                    band = exposure.rescale_intensity(band,
                            in_range=perc, out_range=(0, 255)).astype(np.uint8)
                    new_img.append(band)
                img = np.array(new_img)

            img_path = os.path.join(output_dir, 'jpg', '{i}_{j}.jpg'.format(i=i, j=j))
            image_was_saved = write_image(img, img_path, percentiles=percentiles)
            if image_was_saved:
                saved_windows.append(window)

        write_chips_geojson(saved_windows, crs=ds.crs, output_dir=output_dir)


def write_image(img, path, percentiles=None):
    rgb = np.dstack(img[:3, :, :])
    if exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imsave(path, rgb)
    return True


def write_chips_geojson(windows, *, crs, output_dir):
    _logger.info("Write chips geojson")
    pass


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract chips from a raster file")

    parser.add_argument(
        'raster',
        help='input raster file')
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='size of image tiles, in pixels')
    parser.add_argument(
        '--step-size',
        type=int,
        default=128,
        help='step size (i.e. stride), in pixels')
    parser.add_argument(
        '--contour-shapefile',
        help='contour shapefile')
    parser.add_argument(
	'-o',
        '--output-dir',
        help='output dir', default='.')
    parser.add_argument(
        "--rescale-intensity",
        dest='rescale_intensity',
        default=True,
        action='store_true',
        help="rescale intensity")
    parser.add_argument(
        "--no-rescale-intensity",
        dest='rescale_intensity',
        action='store_false',
        help="do not rescale intensity")
    parser.add_argument(
        "--lower-cut",
        type=int,
        default=2,
        help=
        "lower cut of percentiles for cumulative count in intensity rescaling")
    parser.add_argument(
        "--upper-cut",
        type=int,
        default=98,
        help=
        "upper cut of percentiles for cumulative count in intensity rescaling")

    parser.add_argument(
        "--version",
        action="version",
        version="satproc {ver}".format(ver=__version__))
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    _logger.info("Calculate percentiles")
    percentiles = None
    if args.rescale_intensity:
        percentiles = calculate_percentiles(args.raster,
                lower_cut=args.lower_cut, upper_cut=args.upper_cut)

    _logger.info("Extract chips")
    extract_chips(args.raster,
            size=args.size,
            step_size=args.step_size,
            contour_shapefile=args.contour_shapefile,
            percentiles=percentiles,
            output_dir=args.output_dir)

    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
