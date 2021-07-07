import os
from functools import partial
from glob import glob

import numpy as np
import rasterio

from satproc.utils import map_with_threads


def get_max_prob(p):
    with rasterio.open(p) as src:
        img = src.read()
        return np.max(img)


def filter_chip(src, *, threshold, output_dir):
    if get_max_prob(src) >= threshold:
        dst = os.path.join(output_dir, os.path.basename(src))
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(src, dst)


def filter_by_max_prob(input_dir, output_dir, threshold):
    threshold = round(threshold * 255)
    files = glob(os.path.join(input_dir, "*"))
    worker = partial(filter_chip, output_dir=output_dir, threshold=threshold)
    map_with_threads(files, worker)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter directory of result chips by max. prob threshold"
    )
    parser.add_argument("input_dir", help="results directory")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="apply threshold (between 0 and 1)",
        default=0.5,
    )
    parser.add_argument("-o", "--output-dir", help="output results directory")

    args = parser.parse_args()

    filter_by_max_prob(
        input_dir=args.input_dir, output_dir=args.output_dir, threshold=args.threshold
    )
