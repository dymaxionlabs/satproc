import logging
import rasterio
import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODES = ["gaussian", "median"]


def spatial_filter(*, input_path, output_path, mode="gaussian", size=5):
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
