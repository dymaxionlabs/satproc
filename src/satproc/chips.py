import logging
import os

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box, shape
from shapely.ops import unary_union
from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from satproc.masks import multiband_chip_mask_by_classes, prepare_label_shapes
from satproc.utils import rescale_intensity, sliding_windows, write_chips_geojson

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def extract_chips(
    rasters,
    aoi=None,
    labels=None,
    label_property="class",
    masks={"extent"},
    mask_type="class",
    extent_no_border=False,
    rescale_mode=None,
    rescale_range=None,
    bands=None,
    chip_type="tif",
    write_footprints=False,
    classes=None,
    crs=None,
    skip_existing=True,
    within=False,
    windows_mode="whole_overlap",
    skip_low_contrast=False,
    *,
    size,
    step_size,
    output_dir,
):
    if aoi:
        _logger.info("Prepare AOI shape")
        aoi_poly = prepare_aoi_shape(aoi)
    else:
        aoi_poly = None

    if labels:
        _logger.info("Prepare label shapes")
        polys_dict = prepare_label_shapes(
            labels, mask_type=mask_type, label_property=label_property, classes=classes
        )
    else:
        polys_dict = None

    with logging_redirect_tqdm():
        for raster in tqdm(rasters, desc="Rasters", ascii=True):
            extract_chips_from_raster(
                raster,
                size=size,
                step_size=step_size,
                rescale_mode=rescale_mode,
                rescale_range=rescale_range,
                bands=bands,
                output_dir=output_dir,
                chip_type=chip_type,
                within=within,
                write_footprints=write_footprints,
                crs=crs,
                labels=labels,
                label_property=label_property,
                classes=classes,
                masks=masks,
                mask_type=mask_type,
                extent_no_border=extent_no_border,
                aoi_poly=aoi_poly,
                polys_dict=polys_dict,
                windows_mode=windows_mode,
                skip_existing=skip_existing,
                skip_low_contrast=skip_low_contrast,
            )


def extract_chips_from_raster(
    raster,
    rescale_mode=None,
    rescale_range=None,
    bands=None,
    chip_type="tif",
    write_footprints=False,
    labels=None,
    label_property="class",
    mask_type="class",
    masks={"extent"},
    classes=None,
    crs=None,
    skip_existing=True,
    within=False,
    aoi_poly=None,
    polys_dict=None,
    windows_mode="whole_overlap",
    skip_low_contrast=False,
    extent_no_border=False,
    *,
    size,
    step_size,
    output_dir,
):
    if extent_no_border and "extent" not in masks:
        _logger.warn(
            (
                "You specified `no_border` option but will be ignored "
                "(no `extent` mask specified)"
            )
        )

    if skip_existing:
        _logger.info("Will skip existing files")

    basename, _ = os.path.splitext(os.path.basename(raster))

    with rasterio.open(raster) as ds:
        _logger.info("Raster size: %s", (ds.width, ds.height))

        if any(b > ds.count for b in bands):
            raise RuntimeError(
                f"Raster has {ds.count} bands, "
                f"but you asked to use {bands} band indexes"
            )

        if bands is None:
            bands = list(range(1, min(ds.count, 3) + 1))

        _logger.info("Building windows")
        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(
            sliding_windows(
                win_size, win_step_size, ds.width, ds.height, mode=windows_mode
            )
        )
        _logger.info("Total windows: %d", len(windows))

        _logger.info("Building window shapes")
        window_shapes = [
            box(*rasterio.windows.bounds(w, ds.transform)) for w, _ in windows
        ]
        window_and_shapes = zip(windows, window_shapes)

        # Filter windows by AOI shape
        if aoi_poly:
            _logger.info("Filtering windows by AOI")
            _logger.info('Using "%s" function', "within" if within else "intersects")

            def filter_fn(w, aoi):
                if within:
                    return w.within(aoi)
                else:
                    return w.intersects(aoi)

            window_and_shapes = [
                (w, s) for w, s in window_and_shapes if filter_fn(s, aoi_poly)
            ]
            _logger.info("Total windows after filtering: %d", len(window_and_shapes))

        meta = ds.meta.copy()
        if crs:
            meta["crs"] = CRS.from_string(crs)
        if rescale_mode:
            # If rescaling, set nodata=0 (will rescale to uint8 1-255)
            meta["nodata"] = 0

        chips = []
        for c, ((window, (i, j)), win_shape) in tqdm(
            list(enumerate(window_and_shapes)),
            desc=f"{os.path.basename(raster)} windows",
            ascii=True,
        ):
            _logger.debug("%s %s", window, (i, j))

            name = f"{basename}_{i}_{j}.{chip_type}"
            img_path = os.path.join(output_dir, "images", name)
            mask_paths = {kind: os.path.join(output_dir, kind, name) for kind in masks}

            # Gather list of required files
            required_files = {img_path}
            if labels:
                required_files = required_files | set(mask_paths.values())

            # If all files already exist and we are skipping existing files, continue
            if skip_existing and all(os.path.exists(p) for p in required_files):
                continue

            img = ds.read(window=window)
            img = np.nan_to_num(img)
            img = np.array([img[b - 1, :, :] for b in bands])

            if rescale_mode:
                img = rescale_intensity(img, rescale_mode, rescale_range)

            if chip_type == "tif":
                image_was_saved = write_tif(
                    img,
                    img_path,
                    window=window,
                    meta=meta.copy(),
                    transform=ds.transform,
                    bands=bands,
                    skip_low_contrast=skip_low_contrast,
                )
            else:
                image_was_saved = write_image(
                    img,
                    img_path,
                    skip_low_contrast=skip_low_contrast,
                )

            if image_was_saved:
                chip = (win_shape, (c, i, j))
                chips.append(chip)

                if labels:
                    if mask_type == "class":
                        keys = classes if classes is not None else polys_dict.keys()
                        multiband_chip_mask_by_classes(
                            classes=keys,
                            transform=ds.transform,
                            window=window,
                            polys_dict=polys_dict,
                            metadata=meta,
                            extent_mask_path=mask_paths.get("extent"),
                            boundary_mask_path=mask_paths.get("boundary"),
                            distance_mask_path=mask_paths.get("distance"),
                            label_property=label_property,
                            extent_no_border=extent_no_border,
                        )
                    else:
                        raise RuntimeError(f"mask type '{mask_type}' not supported")

        if write_footprints:
            geojson_path = os.path.join(output_dir, "{}.geojson".format(basename))
            _logger.info("Write chips footprints GeoJSON at %s", geojson_path)
            write_chips_geojson(
                geojson_path,
                chips,
                chip_type=chip_type,
                crs=str(meta["crs"]),
                basename=basename,
            )


def write_image(img, path, *, percentiles=None, skip_low_contrast=False):
    rgb = np.dstack(img[:3, :, :]).astype(np.uint8)
    if skip_low_contrast and exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imsave(path, rgb)
    return True


def write_tif(img, path, *, skip_low_contrast=False, window, meta, transform, bands):
    if skip_low_contrast and exposure.is_low_contrast(img):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta.update(
        {
            "driver": "GTiff",
            "dtype": img.dtype,
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, transform),
            "count": len(bands),
        }
    )
    img = np.array([img[b - 1, :, :] for b in bands])
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(img)
    return True


def get_shape(feature):
    """Get shape geometry from feature

    Parameters
    ----------
    feature : dict
        Feature as read from Fiona

    Returns
    -------
    shapely.geometry.BaseGeometry

    """
    geom = feature["geometry"]
    try:
        return shape(geom)
    except Exception as err:
        _logger.warn("Failed to get shape from feature %s: %s", feature, err)
        return None


def prepare_aoi_shape(aoi):
    with fiona.open(aoi) as src:
        aoi_polys = [get_shape(f) for f in src]
        aoi_polys = [shp for shp in aoi_polys if shp and shp.is_valid]
        aoi_poly = unary_union(aoi_polys)
        return aoi_poly
