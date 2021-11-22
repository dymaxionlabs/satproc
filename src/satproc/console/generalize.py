# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from glob import glob

from satproc import __version__
from satproc.postprocess.generalize import generalize

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
        description="Generalize vector files by simplifying and/or smoothing polygons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_vector", nargs="*", help="input vector file")
    parser.add_argument("--input-dir", help="directory containing input rasters")
    parser.add_argument("-o", "--output-dir", help="output directory")

    parser.add_argument(
        "-tcrs",
        "--target-crs",
        default=None,
        help="reproject to another CRS before processing (EPSG code)",
    )

    parser.add_argument(
        "--simplify",
        choices=["douglas"],
        default="douglas",
        help="simplification method",
    )
    parser.add_argument(
        "--douglas-tolerance",
        type=float,
        default=0.1,
        help="Douglas-Peucker tolerance",
    )

    parser.add_argument(
        "--smooth",
        choices=["chaikin"],
        default=None,
        help="smoothing method",
    )
    parser.add_argument(
        "--chaikin-refinements",
        default=5,
        type=int,
        help="number of Chaikin refinements",
    )

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
        level=loglevel,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args, parser = parse_args(args)
    setup_logging(args.loglevel)

    input_files = []
    if args.input_vector:
        input_files.extend(args.input_vector)
    if args.input_dir:
        files = list(glob(os.path.join(args.input_dir, "*")))
        input_files.extend(files)

    if not input_files:
        raise RuntimeError(
            (
                "No input files found. "
                "You should pass individual input_vector paths, or use --input-dir."
            )
        )
    _logger.info("Num. input files: %d", len(input_files))

    generalize(
        input_files=input_files,
        output_dir=args.output_dir,
        target_crs=args.target_crs,
        simplify=args.simplify,
        douglas_tolerance=args.douglas_tolerance,
        smooth=args.smooth,
        chaikins_refinements=args.chaikin_refinements,
    )


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
