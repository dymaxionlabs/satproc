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

from satproc.build_dataset import build_dataset

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Create dataset from a raster file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("raster", nargs="+", help="input raster file")
    parser.add_argument("dataset", help="input shape or geojson file")
    parser.add_argument(
        "--size", type=int, default=512, help="size of image tiles, in pixels"
    )
    parser.add_argument(
        "--step-size", type=int, default=128, help="step size (i.e. stride), in pixels"
    )
    parser.add_argument("-o", "--output-dir", help="output dir", default=".")

    parser.add_argument(
        "--instance",
        type=bool,
        help="generate a mask from each polygon if true, "
        "else a single mask for each chip",
        default=False,
    )
    parser.add_argument(
        "--type",
        type=str,
        help="dataset output type ['COCO', 'Retinanet']",
        default="COCO",
    )

    parser.add_argument(
        "--label",
        type=str,
        help="annotation label for retinanet dataset",
        default="unknown",
    )

    parser.add_argument("-b", "--bands", nargs="+", type=int, help="RGB band indexes")

    parser.add_argument(
        "--rescale",
        dest="rescale",
        default=True,
        action="store_true",
        help="rescale intensity using percentiles (lower/upper cuts)",
    )
    parser.add_argument(
        "--no-rescale",
        dest="rescale",
        action="store_false",
        help="do not rescale intensity",
    )
    parser.add_argument(
        "--rescale-mode",
        default="percentiles",
        choices=["percentiles", "values"],
        help="choose mode of intensity rescaling",
    )

    parser.add_argument(
        "--lower-cut",
        type=float,
        default=2,
        help="(for 'percentiles' mode) lower cut of percentiles "
        "for cumulative count in intensity rescaling",
    )
    parser.add_argument(
        "--upper-cut",
        type=float,
        default=98,
        help="(for 'percentiles' mode) upper cut of percentiles "
        "for cumulative count in intensity rescaling",
    )

    parser.add_argument(
        "--min",
        type=float,
        help="(for 'values' mode) minimum value in intensity rescaling",
    )
    parser.add_argument(
        "--max",
        type=float,
        help="(for 'values' mode) maximum value in intensity rescaling",
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

    bands = [1, 2, 3] if not args.bands else args.bands

    rescale_mode = args.rescale_mode if args.rescale else None
    if rescale_mode == "percentiles":
        rescale_range = (args.lower_cut, args.upper_cut)
        _logger.info("Rescale intensity with percentiles %s", rescale_range)
    elif rescale_mode == "values":
        rescale_range = (args.min, args.max)
        _logger.info("Rescale intensity with values %s", rescale_range)
    else:
        rescale_range = None
        _logger.info("No rescale intensity")

    _logger.info("Create dataset")
    build_dataset(
        args.dataset,
        args.raster,
        chip_size=args.size,
        step_size=args.step_size,
        output_dir=args.output_dir,
        instance=args.instance,
        type=args.type,
        label=args.label,
        rescale_mode=rescale_mode,
        rescale_range=rescale_range,
        bands=bands,
    )


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
