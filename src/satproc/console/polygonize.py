# -*- coding: utf-8 -*-
import argparse
import logging
import sys

from satproc import __version__
from satproc.postprocess.polygonize import polygonize
from tqdm import tqdm

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
        description="Polygonize results into a single vector file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", help="results directory")
    parser.add_argument("-o", "--output", help="output in GPKG format")
    parser.add_argument("--temp-dir", help="temporary directory")

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

    polygonize(input_dir=args.input_dir,
               output=args.output,
               temp_dir=args.temp_dir)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
