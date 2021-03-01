# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from glob import glob

from satproc import __version__
from satproc.postprocess.polygonize import polygonize
from tqdm import tqdm

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Polygonize raster images into a single vector file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_image", nargs="*", help="input raster")
    parser.add_argument("--input-dir",
                        help="directory containing input rasters")
    parser.add_argument("-o", "--output", help="output in GPKG format")
    parser.add_argument("--temp-dir", help="temporary directory")

    parser.add_argument("-T",
                        "--tile-size",
                        type=int,
                        default=None,
                        help="retile input files with specific tile size")

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

    input_files = []
    if args.input_image:
        input_files.extend(args.input_image)
    if args.input_dir:
        files = list(glob(os.path.join(args.input_dir, '*.tif')))
        input_files.extend(files)

    if not input_files:
        raise RuntimeError(
            "no input files found (you should pass individual input_image paths, or use --input-dir"
        )
    _logger.info("Num. input files: %d", len(input_files))

    polygonize(input_files=input_files,
               output=args.output,
               temp_dir=args.temp_dir,
               tile_size=args.tile_size)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
