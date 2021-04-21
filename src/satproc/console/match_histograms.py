# -*- coding: utf-8 -*-
"""
Apply histogram matching between two rasters.

It manipulates the pixels of an input image so that its histogram matches the
histogram of the reference image. The matching is done independently for each
band, as long as the number of bands is equal in the input image and the
reference.
"""

import argparse
import logging
import sys

from satproc.histogram import match_histograms

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
        description="Match histograms between images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("src", help="input raster")
    parser.add_argument("dst", help="output raster")
    parser.add_argument("-r", "--reference", required=True, help="reference raster")

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

    match_histograms(
        args.src,
        args.dst,
        reference_path=args.reference,
    )


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
