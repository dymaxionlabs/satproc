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
from satproc.masks import make_masks

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
        description="Build masks from chip raster files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("raster", nargs="+", help="input raster file(s)")

    parser.add_argument(
        "--masks",
        "-m",
        nargs="+",
        choices=["extent", "boundary", "distance"],
        default={"extent"},
    )
    parser.add_argument(
        "--mask-type", choices=["single", "class", "instance"], default="class"
    )
    parser.add_argument("--labels", required=True, help="inpul label shapefile")
    parser.add_argument(
        "--label-property",
        default="class",
        help="label property to separate in classes",
    )
    parser.add_argument(
        "--classes", nargs="+", type=str, help="specify classes order in result mask."
    )
    parser.add_argument(
        "--extent-no-border",
        action="store_true",
        help="do not include border in extent mask",
    )
    parser.add_argument("-o", "--output-dir", help="output dir", default=".")

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
    return parser.parse_args(args), parser


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
    args, _parser = parse_args(args)
    setup_logging(args.loglevel)

    make_masks(
        args.raster,
        output_dir=args.output_dir,
        masks=args.masks,
        mask_type=args.mask_type,
        labels=args.labels,
        label_property=args.label_property,
        classes=args.classes,
        extent_no_border=args.extent_no_border,
    )


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
