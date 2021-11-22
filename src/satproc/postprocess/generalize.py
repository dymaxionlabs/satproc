import logging
import os

import fiona
import numpy as np
from pyproj.crs import CRS
from shapely.geometry import Polygon, mapping, shape
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..utils import (
    fiona_crs_from_proj_crs,
    proj_crs_from_fiona_dataset,
    reproject_shape,
)

_logger = logging.getLogger(__name__)


def generalize(
    *,
    input_files,
    output_dir,
    target_crs=None,
    simplify="douglas",
    douglas_tolerance=0.1,
    smooth=None,
    chaikins_refinements=5,
):
    if target_crs:
        target_crs = CRS.from_user_input(target_crs)
        _logger.info("Going to reproject shapes to %s", target_crs.to_string())
        if target_crs.is_geographic:
            _logger.warn(
                (
                    "Target CRS is geographic (%s), so keep in mind "
                    "that distance-related parameters are in degrees. "
                    "Consider reprojecting to a projected local "
                    "coordinate system instead for accurate results."
                ),
                target_crs.to_string(),
            )

    with logging_redirect_tqdm():
        for input_file in tqdm(input_files, ascii=True, desc="Generalize"):
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with fiona.open(input_file) as src:
                proj_crs = proj_crs_from_fiona_dataset(src)
                if not target_crs and proj_crs.is_geographic:
                    _logger.warn(
                        (
                            "File %s has a geographic CRS (%s), so keep in mind "
                            "that distance-related parameters are in degrees. "
                            "Consider reprojecting to a projected local "
                            "coordinate system for accurate results."
                        ),
                        input_file,
                        proj_crs.to_string(),
                    )

                dst_meta = src.meta.copy()
                if target_crs:
                    del dst_meta["crs"]
                    dst_meta["crs_wkt"] = fiona_crs_from_proj_crs(target_crs)

                with fiona.open(output_file, "w", **dst_meta) as dst:
                    for feat in tqdm(
                        src, ascii=True, desc=os.path.basename(input_file)
                    ):
                        shp = shape(feat["geometry"])
                        if target_crs:
                            shp = reproject_shape(shp, proj_crs, target_crs)
                        if simplify == "douglas":
                            shp = shp.simplify(
                                tolerance=douglas_tolerance, preserve_topology=True
                            )
                        if smooth == "chaikin":
                            shp = smooth_chaikin(shp, refinements=chaikins_refinements)
                        feat["geometry"] = mapping(shp)
                        dst.write(feat)


# Based on https://stackoverflow.com/a/47255374/1650058
def smooth_chaikin(shp, refinements=5):
    coords = np.array(shp.exterior.coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return Polygon(coords)
