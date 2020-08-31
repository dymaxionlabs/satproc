import logging
import os

import numpy as np
import rasterio
from rasterio.windows import bounds
from shapely.geometry import shape, box
from shapely.validation import explain_validity
from shapely.ops import transform
from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm

from satproc.utils import (rescale_intensity, sliding_windows,
                           write_chips_geojson)
from rasterio.features import rasterize

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona

__author__ = "Damián Silvani"
__copyright__ = "Damián Silvani"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def mask_from_polygons(polygons, win, mask_path, src, kwargs, image_index,
                       mask_index):
    if polygons:
        mask = rasterize(polygons, (win.height, win.width),
                         transform=rasterio.windows.transform(
                             win, src.transform))
        if mask is None:
            Exception("A empty mask Was generated - Image {} Mask {}".format(
                image_index, mask_index))
    else:
        mask = np.zeros((win.height, win.width), dtype=np.uint8)

    # Write tile
    kwargs.update(dtype=rasterio.uint8, count=1, nodata=0)
    dst_name = '{}/{}_{}.tif'.format(mask_path, image_index, mask_index)
    with rasterio.open(dst_name, 'w', **kwargs) as dst:
        dst.write(mask, 1)

    return mask


def extract_chips(raster,
                  contour_shapefile=None,
                  rescale_mode=None,
                  rescale_range=None,
                  bands=None,
                  type='tif',
                  write_geojson=False,
                  labels=None,
                  label_property='class',
                  mask_type='class',
                  *,
                  size,
                  step_size,
                  output_dir):

    basename, _ = os.path.splitext(os.path.basename(raster))

    if labels:
        blocks = fiona.open(labels) 
        masks_folder = os.path.join(output_dir, "masks")
        os.makedirs(masks_folder, exist_ok=True)
        if mask_type == 'class':
            polys_dict = {}
            for block in blocks:
                if label_property in block['properties']:
                    c = block['properties'][label_property]
                    geom = shape(block['geometry'])
                    if c in polys_dict:
                        polys_dict[c].append(geom) 
                    else:
                        polys_dict[c] = [geom]

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
        for c, (window, (i, j)) in tqdm(list(enumerate(windows))):
            _logger.debug("%s %s", window, (i, j))
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            img = np.array([img[b - 1, :, :] for b in bands])

            if rescale_mode:
                img = rescale_intensity(img, rescale_mode, rescale_range)

            img_path = os.path.join(output_dir, f"{basename}_{i}_{j}.{type}")

            if type == 'tif':
                image_was_saved = write_tif(img,
                                            img_path,
                                            window=window,
                                            meta=meta,
                                            transform=ds.transform)
            else:
                image_was_saved = write_image(img, img_path)

            if image_was_saved:
                chip_shape = box(
                    *rasterio.windows.bounds(window, ds.transform))
                chip = (chip_shape, (c, i, j))
                chips.append(chip)

                if labels:
                    if mask_type == 'class':
                        for key, class_blocks in polys_dict.items():
                            intersect_polys = []
                            for s in class_blocks:
                                if s.is_valid:
                                    i = chip_shape.intersection(s)
                                    if i:
                                        intersect_polys.append(i)
                                else:
                                    _logger.warn(f"Invalid geometry {explain_validity(s)}")
                            if len(intersect_polys) > 0:
                                mask_from_polygons(
                                    intersect_polys, 
                                    window,
                                    masks_folder,
                                    ds, 
                                    ds.meta.copy(), 
                                    f"{i}_{j}", 
                                    key
                                )



        if write_geojson:
            geojson_path = os.path.join(output_dir,
                                        "{}.geojson".format(basename))
            write_chips_geojson(geojson_path,
                                chips,
                                type=type,
                                crs=str(ds.crs),
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
        'height': window.height,
        'width': window.width,
        'transform': rasterio.windows.transform(window, transform)
    })
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(img)
    return True
