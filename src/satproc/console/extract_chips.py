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

import os
import math
import rasterio
import numpy as np
import pyproj
import json
from tqdm import tqdm
from rasterio.windows import Window, bounds
from skimage.io import imsave
from skimage import exposure
from shapely.geometry import box, mapping
from shapely.ops import transform
from functools import partial

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona

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
            img = np.nan_to_num(img)
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


def extract_chips(raster, contour_shapefile=None, percentiles=None, bands=[1, 2, 3], *, size,
        step_size, output_dir):
    with rasterio.open(raster) as ds:
        _logger.info('Raster size: %s', (ds.width, ds.height))

        if ds.count < 3:
            raise RuntimeError("Raster must have 3 bands corresponding to RGB channels")

        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(sliding_windows(win_size, win_step_size, ds.width, ds.height))
        chips = []

        for c, (window, (i, j)) in tqdm(list(enumerate(windows))):
            _logger.debug("%s %s", window, (i, j))
            img = ds.read(window=window)
            img = np.nan_to_num(img)

            if percentiles:
                new_img = []
                for k, perc in enumerate(percentiles):
                    band = img[k, :, :]
                    band = exposure.rescale_intensity(band,
                            in_range=perc, out_range=(0, 255)).astype(np.uint8)
                    new_img.append(band)
                img = np.array(new_img)
            else:
                img = np.array([img[b-1, :, :] for b in bands])

            img_path = os.path.join(output_dir, '{i}_{j}.jpg'.format(i=i, j=j))
            image_was_saved = write_image(img, img_path)
            if image_was_saved:
                chip_shape = box(*bounds(window, ds.transform))
                chip = (chip_shape, (c, i, j))
                chips.append(chip)

        write_chips_geojson(chips, crs=str(ds.crs), output_dir=output_dir)


def write_image(img, path, percentiles=None):
    rgb = np.dstack(img[:3, :, :]).astype(np.uint8)
    if exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        imsave(path, rgb)
    return True


def write_chips_geojson(chip_pairs, *, crs, output_dir):
    if not chip_pairs:
        _logger.warn("No chips to save")
        return

    _logger.info("Write chips geojson")
    output_path = os.path.join(output_dir, 'chips.geojson')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        d = { 'type': 'FeatureCollection',
              'features': [] }
        for i, (chip, (fi, xi, yi)) in enumerate(chip_pairs):
            # Shapes will be stored in EPSG:4326 projection
            if crs != 'EPSG:4326':
                project = partial(pyproj.transform,
                        pyproj.Proj(crs),
                        pyproj.Proj('EPSG:4326'))
                chip_wgs = transform(project, chip)
            else:
                chip_wgs = chip
            filename = '{x}_{y}.jpg'.format(x=xi, y=yi)
            feature = { 'type': 'Feature',
                        'geometry': mapping(chip_wgs),
                        'properties': { 'id': i, 'x': xi, 'y': yi, 'filename': filename } }
            d['features'].append(feature)
        f.write(json.dumps(d))


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
        '-b',
        '--bands',
        nargs='+',
        type=int,
        help="RGB band indexes")

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

    percentiles = None
    if args.rescale_intensity:
        _logger.info("Calculate percentiles")
        percentiles = ((0.0, 0.17889499664306618), (0.0, 0.1965800046920776), (0.0, 0.25697999894618984), (0.0, 0.29840499460697156))
        #percentiles = calculate_percentiles(args.raster,
                #lower_cut=args.lower_cut, upper_cut=args.upper_cut)

    _logger.info("Extract chips")
    extract_chips(args.raster,
            size=args.size,
            step_size=args.step_size,
            contour_shapefile=args.contour_shapefile,
            percentiles=percentiles,
            bands=args.bands,
            output_dir=args.output_dir)

    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
