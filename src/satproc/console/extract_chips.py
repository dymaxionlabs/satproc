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
from satproc.utils import calculate_percentiles

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
    parser = argparse.ArgumentParser(description="Extract chips from a raster file")

    parser.add_argument("raster", help="input raster file")
    parser.add_argument(
        "--size", type=int, default=256, help="size of image tiles, in pixels"
    )
    parser.add_argument(
        "--step-size", type=int, default=128, help="step size (i.e. stride), in pixels"
    )
    parser.add_argument("--contour-shapefile", help="contour shapefile")
    parser.add_argument("-o", "--output-dir", help="output dir", default=".")
    parser.add_argument(
        "--rescale-intensity",
        dest="rescale_intensity",
        default=True,
        action="store_true",
        help="rescale intensity",
    )
    parser.add_argument(
        "--no-rescale-intensity",
        dest="rescale_intensity",
        action="store_false",
        help="do not rescale intensity",
    )
    parser.add_argument(
        "--lower-cut",
        type=int,
        default=2,
        help="lower cut of percentiles for cumulative count in intensity rescaling",
    )
    parser.add_argument(
        "--upper-cut",
        type=int,
        default=98,
        help="upper cut of percentiles for cumulative count in intensity rescaling",
    )
    parser.add_argument("-b", "--bands", nargs="+", type=int, help="RGB band indexes")

    parser.add_argument(
        "--version", action="version", version="satproc {ver}".format(ver=__version__)
    )
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
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


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
        percentiles = calculate_percentiles(
            args.raster, lower_cut=args.lower_cut, upper_cut=args.upper_cut
        )

    _logger.info("Extract chips")
    extract_chips(
        args.raster,
        size=args.size,
        step_size=args.step_size,
        contour_shapefile=args.contour_shapefile,
        percentiles=percentiles,
        bands=args.bands,
        output_dir=args.output_dir,
    )

    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
