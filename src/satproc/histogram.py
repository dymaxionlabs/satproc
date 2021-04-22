import logging

import numpy as np
import rasterio
from skimage import exposure
from tqdm import tqdm

from satproc.utils import sliding_windows

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)

# TODO add win size and step size


def read_window(ds, window):
    """Read from a rasterio dataset using a window

    NaNs are coerced to zero.

    Parameters
    ----------
    ds : rasterio.Dataset
        input dataset
    window : rasterio.windows.Window
        window to read from

    Returns
    -------
    numpy.ndarray
        image data on window

    """

    img = ds.read(window=window)
    img = np.nan_to_num(img)
    return np.dstack(img)


def write_window(img, ds, window):
    """Write array to raster on window

    Parameters
    ----------
    img : numpy.ndarray
        image array
    ds : rasterio.Dataset
        dataset to write to (must be opened with write access)
    window : rasterio.windows.Window
        window to write to

    Returns
    -------
    None

    """
    new_img = np.array([img[:, :, i] for i in range(img.shape[2])])
    ds.write(new_img, window=window)


def match_histograms(src_path, dst_path, size=128, step_size=128, *, reference_path):
    """Match histograms of an image using another one as reference

    Parameters
    ----------
    src_path : str
        path to input raster
    dst_path : str
        path to output raster
    size : int
        size of windows
    step_size : int
        step size, when sliding windows
    reference_path : str
        path to the reference raster

    Returns
    -------
    None

    """
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        windows = list(
            sliding_windows(
                (size, size), (step_size, step_size), src.width, src.height, whole=False
            )
        )

        with rasterio.open(reference_path) as ref:
            with rasterio.open(dst_path, "w", **profile) as dst:
                for c, (win, (i, j)) in tqdm(list(enumerate(windows))):
                    _logger.debug("%s %s", win, (i, j))

                    img = read_window(src, win)
                    ref_img = read_window(ref, win)

                    matched_img = exposure.match_histograms(
                        img, ref_img, multichannel=True
                    )
                    write_window(matched_img, dst, win)
