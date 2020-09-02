import logging
import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import bounds
from rasterio.warp import calculate_default_transform
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from shapely.validation import explain_validity
from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm

from satproc.utils import (rescale_intensity, sliding_windows,
                           write_chips_geojson)

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def mask_from_polygons(polygons, *, win, t):
    transform = rasterio.windows.transform(win, t)
    if polygons is None or len(polygons) == 0:
        mask = np.zeros((win.height, win.width), dtype=np.uint8)
    else:
        mask = rasterize(polygons, (win.height, win.width),
                    default_value=255,
                    transform=transform)
    return mask


def extract_chips(raster,
                  aoi=None,
                  rescale_mode=None,
                  rescale_range=None,
                  bands=None,
                  type='tif',
                  write_geojson=False,
                  labels=None,
                  label_property='class',
                  mask_type='class',
                  classes=None,
                  crs=None,
                  *,
                  size,
                  step_size,
                  output_dir):

    basename, _ = os.path.splitext(os.path.basename(raster))

    masks_folder = os.path.join(output_dir, "masks")
    image_folder = os.path.join(output_dir, "images")

    if aoi:
        with fiona.open(aoi) as src:
            aoi_polys = [shape(f['geometry']) for f in src]
            aoi_polys = [shp for shp in aoi_polys if shp.is_valid]
            aoi_poly = unary_union(aoi_polys)

    if labels and mask_type == 'class':
        with fiona.open(labels) as blocks:
            polys_dict = {}
            for block in blocks:
                if label_property in block['properties']:
                    c = block['properties'][label_property]
                    geom = shape(block['geometry'])
                    if c in polys_dict:
                        polys_dict[c].append(geom)
                    else:
                        polys_dict[c] = [geom]
        if classes:
            for c in classes:
                if c not in polys_dict:
                    polys_dict[c] = []
                    _logger.warn(f"No features of class '{c}' found. Will generate empty masks.")

    with rasterio.open(raster) as ds:
        _logger.info("Raster size: %s", (ds.width, ds.height))

        if any(b > ds.count for b in bands):
            raise RuntimeError(
                f"Raster has {ds.count} bands, but you asked to use {bands} band indexes"
            )

        if bands is None:
            bands = list(range(1, min(ds.count, 3) + 1))

        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(
            sliding_windows(win_size,
                            win_step_size,
                            ds.width,
                            ds.height,
                            whole=True))
        chips = []

        meta = ds.meta.copy()
        if crs:
            meta['crs'] = CRS.from_string(crs)

        for c, (window, (i, j)) in tqdm(list(enumerate(windows))):
            _logger.debug("%s %s", window, (i, j))

            chip_shape = box(*rasterio.windows.bounds(window, ds.transform))

            if aoi and not chip_shape.intersects(aoi_poly):
                continue

            img = ds.read(window=window)
            img = np.nan_to_num(img)
            img = np.array([img[b - 1, :, :] for b in bands])

            if rescale_mode:
                img = rescale_intensity(img, rescale_mode, rescale_range)

            img_path = os.path.join(image_folder, f"{basename}_{i}_{j}.{type}")

            if type == 'tif':
                image_was_saved = write_tif(img,
                                            img_path,
                                            window=window,
                                            meta=meta.copy(),
                                            transform=ds.transform)
            else:
                image_was_saved = write_image(img, img_path)

            if image_was_saved:
                chip = (chip_shape, (c, i, j))
                chips.append(chip)

                if labels:
                    if mask_type == 'class':
                        multi_band_mask = []
                        t = ds.transform
                        keys = classes if classes is not None else polys_dict.keys()
                        for k in keys:
                            intersect_polys = []
                            for s in polys_dict[k]:
                                if s.is_valid:
                                    intersection = chip_shape.intersection(s)
                                    if intersection:
                                        intersect_polys.append(intersection)
                                else:
                                    _logger.warn(
                                        f"Invalid geometry {explain_validity(s)}"
                                    )
                            multi_band_mask.append(mask_from_polygons(intersect_polys, win=window, t=t))

                        kwargs = meta.copy()
                        kwargs.update(driver='GTiff',
                                      dtype=rasterio.uint8,
                                      count=len(multi_band_mask),
                                      nodata=0,
                                      transform=rasterio.windows.transform(window, t),
                                      width=window.width,
                                      height=window.height)

                        dst_name = os.path.join(masks_folder, f"{basename}_{i}_{j}.tif")
                        os.makedirs(os.path.dirname(dst_name), exist_ok=True)
                        with rasterio.open(dst_name, 'w', **kwargs) as dst:
                            for i in range(len(multi_band_mask)):
                                dst.write(multi_band_mask[i], i+1)

        if write_geojson:
            geojson_path = os.path.join(output_dir,
                                        "{}.geojson".format(basename))
            write_chips_geojson(geojson_path,
                                chips,
                                type=type,
                                crs=str(meta['crs']),
                                basename=basename)


def write_image(img, path, percentiles=None):
    rgb = np.dstack(img[:3, :, :]).astype(np.uint8)
    if exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        imsave(path, rgb)
    return True


def write_tif(img, path, *, window, meta, transform):
    if exposure.is_low_contrast(img):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, transform)
    })
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(img)
    return True
