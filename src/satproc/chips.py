import logging
import os

# Workaround: Load fiona at the end to avoid segfault on box (???)
import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import box, shape
from shapely.ops import unary_union
from skimage import exposure
from skimage.io import imsave
from tqdm import tqdm

from satproc.utils import rescale_intensity, sliding_windows, write_chips_geojson

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


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


def mask_from_polygons(polygons, *, win, t):
    """Generate a binary mask array from a set of polygon

    Parameters
    ----------
    polygons : List[Union[Polygon, MultiPolygon]]
        list of polygon or multipolygon geometries
    win : rasterio.windows.Window
        window
    t : rasterio.transform.Affine
        affine transform

    Returns
    -------
    numpy.ndarray

    """
    transform = rasterio.windows.transform(win, t)
    if polygons is None or len(polygons) == 0:
        mask = np.zeros((win.height, win.width), dtype=np.uint8)
    else:
        mask = rasterize(
            polygons, (win.height, win.width), default_value=255, transform=transform
        )
    return mask


def multiband_chip_mask_by_classes(
    classes,
    transform,
    window,
    mask_path,
    label_property,
    polys_dict=None,
    label_path=None,
    window_shape=None,
    metadata={},
):
    multi_band_mask = []
    if polys_dict is None and label_path is not None:
        polys_dict = classify_polygons(label_path, label_property, classes)
    if window_shape is None:
        window_shape = box(*rasterio.windows.bounds(window, transform))

    for k in classes:
        multi_band_mask.append(
            mask_from_polygons(polys_dict[k], win=window, t=transform)
        )

    kwargs = metadata.copy()
    kwargs.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=len(multi_band_mask),
        nodata=0,
        transform=rasterio.windows.transform(window, transform),
        width=window.width,
        height=window.height,
    )

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    with rasterio.open(mask_path, "w", **kwargs) as dst:
        for i in range(len(multi_band_mask)):
            dst.write(multi_band_mask[i], i + 1)


def classify_polygons(labels, label_property, classes):
    with fiona.open(labels) as blocks:
        polys_dict = {}
        for block in blocks:
            if label_property in block["properties"]:
                c = str(block["properties"][label_property])
                try:
                    geom = shape(block["geometry"])
                except RuntimeError:
                    _logger.warning(
                        "Failed to get geometry shape for feature: %s", block
                    )
                    continue
                if c in polys_dict:
                    polys_dict[c].append(geom)
                else:
                    polys_dict[c] = [geom]
    if classes:
        for c in classes:
            if c not in polys_dict:
                polys_dict[c] = []
                _logger.warn(
                    f"No features of class '{c}' found. Will generate empty masks."
                )
    return polys_dict


def prepare_aoi_shape(aoi):
    with fiona.open(aoi) as src:
        aoi_polys = [get_shape(f) for f in src]
        aoi_polys = [shp for shp in aoi_polys if shp and shp.is_valid]
        aoi_poly = unary_union(aoi_polys)
        return aoi_poly


def prepare_label_shapes(
    labels, mask_type="class", label_property="class", classes=None
):
    if mask_type == "class":
        polys_dict = classify_polygons(labels, label_property, classes)
        return polys_dict
    else:
        raise RuntimeError(f"mask type '{mask_type}' not supported")


def extract_chips(
    rasters,
    aoi=None,
    labels=None,
    label_property="class",
    mask_type="class",
    rescale_mode=None,
    rescale_range=None,
    bands=None,
    type="tif",
    write_geojson=False,
    classes=None,
    crs=None,
    skip_existing=True,
    within=False,
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

    for raster in tqdm(rasters):
        extract_chips_from_raster(
            raster,
            size=size,
            step_size=step_size,
            rescale_mode=rescale_mode,
            rescale_range=rescale_range,
            bands=bands,
            output_dir=output_dir,
            type=type,
            within=within,
            write_geojson=write_geojson,
            crs=crs,
            labels=labels,
            label_property=label_property,
            classes=classes,
            mask_type=mask_type,
            aoi_poly=aoi_poly,
            polys_dict=polys_dict,
            skip_existing=skip_existing,
        )


def extract_chips_from_raster(
    raster,
    rescale_mode=None,
    rescale_range=None,
    bands=None,
    type="tif",
    write_geojson=False,
    labels=None,
    label_property="class",
    mask_type="class",
    classes=None,
    crs=None,
    skip_existing=True,
    within=False,
    aoi_poly=None,
    polys_dict=None,
    *,
    size,
    step_size,
    output_dir,
):

    if skip_existing:
        _logger.info("Will skip existing files")

    basename, _ = os.path.splitext(os.path.basename(raster))

    masks_folder = os.path.join(output_dir, "masks")
    image_folder = os.path.join(output_dir, "images")

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
            sliding_windows(win_size, win_step_size, ds.width, ds.height, whole=True)
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
            list(enumerate(window_and_shapes))
        ):
            _logger.debug("%s %s", window, (i, j))

            img_path = os.path.join(image_folder, f"{basename}_{i}_{j}.{type}")
            mask_path = os.path.join(masks_folder, f"{basename}_{i}_{j}.{type}")
            if (
                skip_existing
                and os.path.exists(img_path)
                and (not labels or os.path.exists(mask_path))
            ):
                continue

            img = ds.read(window=window)
            img = np.nan_to_num(img)
            img = np.array([img[b - 1, :, :] for b in bands])

            if rescale_mode:
                img = rescale_intensity(img, rescale_mode, rescale_range)

            if type == "tif":
                image_was_saved = write_tif(
                    img,
                    img_path,
                    window=window,
                    meta=meta.copy(),
                    transform=ds.transform,
                    bands=bands,
                )
            else:
                image_was_saved = write_image(img, img_path)

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
                            window_shape=win_shape,
                            polys_dict=polys_dict,
                            metadata=meta,
                            mask_path=mask_path,
                            label_property=label_property,
                        )

        if write_geojson:
            geojson_path = os.path.join(output_dir, "{}.geojson".format(basename))
            _logger.info("Write chips geojson at %s", geojson_path)
            write_chips_geojson(
                geojson_path, chips, type=type, crs=str(meta["crs"]), basename=basename
            )


def write_image(img, path, percentiles=None):
    rgb = np.dstack(img[:3, :, :]).astype(np.uint8)
    if exposure.is_low_contrast(rgb):
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        imsave(path, rgb)
    return True


def write_tif(img, path, *, window, meta, transform, bands):
    if exposure.is_low_contrast(img):
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
