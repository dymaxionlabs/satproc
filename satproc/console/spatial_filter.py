# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import os
from glob import glob

from satproc import __version__
from satproc.postprocess.spatial_filter import MODES, spatial_filter

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def check_kernel_size(value):
    ivalue = int(value)
    if ivalue % 2 == 0:
        raise argparse.ArgumentTypeError("must be an odd number")
    return ivalue


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Apply a spatial filter (median, gaussian) to an image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="satproc {ver}".format(ver=__version__),
    )

    parser.add_argument("input", nargs="*", help="input raster")
    parser.add_argument("--input-dir", help="directory containing input rasters")
    parser.add_argument("--output", "-o", help="output raster")
    parser.add_argument("--mode", "-m", choices=MODES, default="gaussian", help="Filter mode")
    parser.add_argument("--size", "-s", type=check_kernel_size, default=5, help="Size of the filter")
    parser.add_argument(
        "--merge",
        dest="merge",
        default=True,
        action="store_true",
        help="merge all input files into a single virtual raster before filtering",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        default=False,
        action="store_false",
        help="do not merge all input files into a single virtual raster",
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

    input_paths = []
    if args.input:
        input_paths.extend(args.input)
    if args.input_dir:
        files = list(glob(os.path.join(args.input_dir, "*.tif")))
        input_paths.extend(files)

    if not input_paths:
        raise RuntimeError(
            (
                "No input files found. "
                "You should pass individual input_image paths, or use --input-dir."
            )
        )
    _logger.info("Num. input files: %d", len(input_paths))

    spatial_filter(input_paths=input_paths, output_path=args.output, mode=args.mode, size=args.size)


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
