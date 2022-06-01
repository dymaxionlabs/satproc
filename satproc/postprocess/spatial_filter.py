import logging
import tempfile
import os

import rasterio
import cv2
import numpy as np
from tqdm import tqdm

from satproc.utils import build_virtual_raster

_logger = logging.getLogger(__name__)

MODES = ["gaussian", "median"]


def spatial_filter(*, input_paths, output_path, mode="gaussian", size=5, merge=True):
    if merge:
        temp_dir = tempfile.TemporaryDirectory()
        vrt_path = os.path.join(temp_dir.name, "merged.vrt")
        _logger.info(f"Merge input images into a virtual raster {vrt_path}")
        build_virtual_raster(input_paths, vrt_path)
        input_paths = [vrt_path]

    for input_path in tqdm(input_paths, desc="Apply filter", ascii=True):
        with rasterio.open(input_path) as src:
            if size > src.width or size > src.height:
                raise ValueError(f"Kernel size {size} is larger than image size ({src.width}x{src.height})")

            profile = src.profile.copy()
            profile.update(driver="GTiff")

            with rasterio.open(output_path, "w", **profile) as dst:
                for b in range(1, src.count + 1):
                    img = src.read(b)
                    if mode == "gaussian":
                        new_img = cv2.GaussianBlur(img, (size, size), 0)
                    elif mode == "median":
                        new_img = cv2.medianBlur(img, size)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
                    dst.write(new_img, b)
