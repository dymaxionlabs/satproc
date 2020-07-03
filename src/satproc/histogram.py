import logging

import rasterio
import numpy as np
from tqdm import tqdm
from skimage import exposure
from satproc.utils import sliding_windows

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)

# TODO add win size and step size


def read_window(ds, window):
    img = ds.read(window=window)
    img = np.nan_to_num(img)
    return np.dstack(img)


def write_window(img, ds, window):
    new_img = np.array([img[:, :, i] for i in range(img.shape[2])])
    ds.write(new_img, window=window)


def match_histograms(src_path,
                     dst_path,
                     size=128,
                     step_size=128,
                     *,
                     reference_path):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        windows = list(
            sliding_windows((size, size), (step_size, step_size),
                            src.width,
                            src.height,
                            whole=False))

        with rasterio.open(reference_path) as ref:
            with rasterio.open(dst_path, 'w', **profile) as dst:
                for c, (win, (i, j)) in tqdm(list(enumerate(windows))):
                    _logger.debug("%s %s", win, (i, j))

                    img = read_window(src, win)
                    ref_img = read_window(ref, win)

                    matched_img = exposure.match_histograms(img,
                                                            ref_img,
                                                            multichannel=True)
                    write_window(matched_img, dst, win)
