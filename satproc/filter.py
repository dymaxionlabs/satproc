import logging
import os
from functools import partial
from glob import glob

import numpy as np
import rasterio

from satproc.utils import map_with_threads

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


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
