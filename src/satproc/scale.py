import logging

import numpy as np
import rasterio
from tqdm import tqdm

from satproc.utils import sliding_windows

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def get_min_max(img, window_size=512):
    """Return minimum and maximum values on array, in blocks

    Parameters
    ----------
    img : numpy.ndarray
        image array
    window_size : int
        size of window (default: 512)

    Returns
    -------
    Tuple[float, float]
        minimum and maximum values

    """
    mins, maxs = [], []
    with rasterio.open(img) as src:
        win_size = (window_size, window_size)
        windows = list(sliding_windows(win_size, win_size, src.width, src.height))
        for _, (window, (i, j)) in tqdm(list(enumerate(windows))):
            img = src.read(window=window)
            mins.append(np.array([np.nanmin(img[i, :, :]) for i in range(src.count)]))
            maxs.append(np.array([np.nanmax(img[i, :, :]) for i in range(src.count)]))
    min_value = np.nanmin(np.array(mins), axis=0)
    max_value = np.nanmax(np.array(maxs), axis=0)
    return min_value, max_value


def minmax_scale(img, *, min_values, max_values):
    """
    Scale bands of image separately, to range 0..1

    Parameters
    ----------
    img : numpy.ndarray
        image array
    min_values : List[float]
        minimum values for each band
    max_values : List[float]
        maximum values for each band

    Returns
    -------
    numpy.ndarray
        rescaled image

    """
    n_bands = img.shape[0]
    return np.array(
        [
            (img[i, :, :] - min_values[i]) / (max_values[i] - min_values[i])
            for i in range(n_bands)
        ]
    )


def scale(input_img, output_img, window_size=512):
    """
    Read a raster, rescale each band with min-max values, and save as another raster

    Parameters
    ----------
    input_img : str
        path to input image
    output_img : str
        path to output image
    window_size : int
        size of window

    Returns
    -------
    None

    """
    min_values, max_values = get_min_max(input_img, window_size=window_size)

    with rasterio.open(input_img) as src:
        win_size = (window_size, window_size)
        windows = list(sliding_windows(win_size, win_size, src.width, src.height))
        profile = src.profile.copy()
        profile.update(dtype=np.float32)
        with rasterio.open(output_img, "w", **profile) as dst:
            for _, (window, (i, j)) in tqdm(list(enumerate(windows))):
                _logger.debug("%s %s", window, (i, j))
                img = src.read(window=window)
                new_img = minmax_scale(
                    img, min_values=min_values, max_values=max_values
                )
                dst.write(new_img, window=window)
