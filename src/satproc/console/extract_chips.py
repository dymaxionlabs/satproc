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
import logging
import sys

from satproc import __version__
from satproc.chips import extract_chips
from satproc.utils import get_raster_band_count

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract chips from a raster file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("raster", help="input raster file")
    parser.add_argument("--size",
                        type=int,
                        default=256,
                        help="size of image tiles, in pixels")
    parser.add_argument("--step-size",
                        type=int,
                        default=128,
                        help="step size (i.e. stride), in pixels")
    parser.add_argument("--contour-shapefile", help="contour shapefile")
    parser.add_argument("-o", "--output-dir", help="output dir", default=".")

    parser.add_argument(
        "--rescale",
        dest="rescale",
        default=False,
        action="store_true",
        help="rescale intensity using percentiles (lower/upper cuts)",
    )
    parser.add_argument(
        "--no-rescale",
        dest="rescale",
        action="store_false",
        help="do not rescale intensity",
    )

    parser.add_argument("--rescale-mode",
                        default="percentiles",
                        choices=["percentiles", "values"],
                        help="choose mode of intensity rescaling")

    parser.add_argument(
        "--lower-cut",
        type=float,
        default=2,
        help=
        "(for 'percentiles' mode) lower cut of percentiles for cumulative count in intensity rescaling",
    )
    parser.add_argument(
        "--upper-cut",
        type=float,
        default=98,
        help=
        "(for 'percentiles' mode) upper cut of percentiles for cumulative count in intensity rescaling",
    )

    parser.add_argument(
        '--min',
        type=float,
        help="(for 'values' mode) minimum value in intensity rescaling")
    parser.add_argument(
        '--max',
        type=float,
        help="(for 'values' mode) maximum value in intensity rescaling")

    parser.add_argument(
        "-b",
        "--bands",
        nargs="+",
        type=int,
        help=
        "specify band indexes. If type is 'jpg', defaults to (1, 2, 3). If type is 'tif', defaults to the total band count."
    )
    parser.add_argument("-t",
                        "--type",
                        help="output chip format",
                        choices=['jpg', 'tif'],
                        default='tif')

    parser.add_argument("--write-geojson",
                        help="write a GeoJSON file of chip polygons",
                        dest="write_geojson",
                        action="store_true",
                        default=False)
    parser.add_argument("--no-write-geojson",
                        help="do not write a GeoJSON file of chip polygons",
                        dest="write_geojson",
                        action="store_false")

    parser.add_argument("--version",
                        action="version",
                        version="satproc {ver}".format(ver=__version__))
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args), parser


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args, parser = parse_args(args)
    setup_logging(args.loglevel)

    bands = args.bands
    if not bands:
        if args.type == 'jpg':
            bands = [1, 2, 3]
        else:
            bands = list(range(1, get_raster_band_count(args.raster) + 1))

    if type == 'jpg' and len(bands) != 3:
        parser.error(
            f"--type is jpg, but --bands does not have 3 band indexes: {bands}"
        )

    rescale_mode = args.rescale_mode if args.rescale else None
    if rescale_mode == 'percentiles':
        rescale_range = (args.lower_cut, args.upper_cut)
        _logger.info("Rescale intensity with percentiles %s", rescale_range)
    elif rescale_mode == 'values':
        rescale_range = (args.min, args.max)
        _logger.info("Rescale intensity with values %s", rescale_range)
    else:
        rescale_range = None
        _logger.info("No rescale intensity")

    _logger.info("Extract chips")
    extract_chips(args.raster,
                  size=args.size,
                  step_size=args.step_size,
                  contour_shapefile=args.contour_shapefile,
                  rescale_mode=rescale_mode,
                  rescale_range=rescale_range,
                  bands=bands,
                  output_dir=args.output_dir,
                  type=args.type,
                  write_geojson=args.write_geojson)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
