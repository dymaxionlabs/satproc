import logging
import os
from collections import defaultdict

import cv2
import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "Apache-2.0"

_logger = logging.getLogger(__name__)


def make_masks(
    rasters,
    *,
    output_dir,
    labels,
    label_property="class",
    classes=None,
    mask_type="class",
    masks={"extent"},
    extent_no_border=False,
):
    if mask_type not in ("single", "class"):
        raise RuntimeError(f"mask type '{mask_type}' not implemented")

    # If mask type is single, there is a single class
    if mask_type == "single":
        classes = None

    if extent_no_border and "extent" not in masks:
        _logger.warn(
            (
                "You specified `no_border` option but will be ignored "
                "(no `extent` mask specified)"
            )
        )

    polys_dict = classify_polygons(labels, label_property, classes)

    with logging_redirect_tqdm():
        for raster in tqdm(rasters, ascii=True, desc="Rasters"):
            name, _ = os.path.splitext(os.path.basename(raster))
            with rasterio.open(raster) as src:
                profile = src.profile.copy()
                transform = src.transform
                b = src.bounds

                # Calculate window from image bounds
                window = rasterio.windows.from_bounds(
                    b.left,
                    b.bottom,
                    b.right,
                    b.top,
                    src.transform,
                    src.height,
                    src.width,
                )

            paths = {
                kind: os.path.join(output_dir, kind, f"{name}.tif") for kind in masks
            }
            keys = classes if classes is not None else polys_dict.keys()

            masks = multiband_chip_mask_by_classes(
                classes=keys,
                transform=transform,
                window=window,
                polys_dict=polys_dict,
                label_property=label_property,
                extent_no_border=extent_no_border,
                extent_mask_path=paths.get("extent"),
                boundary_mask_path=paths.get("boundary"),
                distance_mask_path=paths.get("distance"),
            )

            write_window_masks(
                masks, window=window, metadata=profile, transform=transform
            )


def multiband_chip_mask_by_classes(
    classes,
    transform,
    window,
    label_property,
    polys_dict=None,
    label_path=None,
    extent_mask_path=None,
    boundary_mask_path=None,
    distance_mask_path=None,
    extent_no_border=False,
):
    mb_extent, mb_boundary, mb_distance = [], [], []

    if polys_dict is None and label_path is not None:
        polys_dict = classify_polygons(label_path, label_property, classes)

    for k in classes:
        polys = polys_dict.get(k, [])
        extent_mask, bound_mask, dist_mask = mask_from_polygons(
            polys,
            win=window,
            t=transform,
            extent_no_border=extent_no_border,
            boundary_mask=boundary_mask_path,
            distance_mask=distance_mask_path,
        )
        mb_extent.append(extent_mask)
        mb_boundary.append(bound_mask)
        mb_distance.append(dist_mask)

    masks = zip(
        [mb_extent, mb_boundary, mb_distance],
        [extent_mask_path, boundary_mask_path, distance_mask_path],
    )
    masks = [(path, mask) for mask, path in masks if path is not None]

    return masks


def all_masks_empty(masks):
    """Check if all masks are empty"""
    for _, mask in masks:
        for arr in mask:
            if np.any(arr):
                return False
    return True


def write_window_masks(masks, *, window, metadata, transform):
    """Write window masks to files"""
    for path, mask in masks:
        kwargs = metadata.copy()
        kwargs.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=len(mask),
            nodata=0,
            transform=rasterio.windows.transform(window, transform),
            width=window.width,
            height=window.height,
        )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with rasterio.open(path, "w", **kwargs) as dst:
            for i in range(len(mask)):
                dst.write(mask[i], i + 1)


def mask_from_polygons(
    polygons,
    *,
    win,
    t,
    extent_no_border=True,
    boundary_mask=None,
    distance_mask=None,
):
    """Generate a binary mask array from a set of polygon

    It can also generate a distance transform mask

    Parameters
    ----------
    polygons : List[Union[Polygon, MultiPolygon]]
        list of polygon or multipolygon geometries
    win : rasterio.windows.Window
        window
    t : rasterio.transform.Affine
        affine transform
    extent_no_border : bool
        if True, the extent mask will not include the border of the polygon
    boundary_mask : bool
        whether to generate boundary (edges) mask
    distance_mask : bool
        whether to generate a distance mask

    Returns
    -------
    numpy.ndarray

    """
    transform = rasterio.windows.transform(win, t)
    win_shape = (int(win.height), int(win.width))
    bound_mask, dist_mask = None, None

    if polygons is None or len(polygons) == 0:
        mask = np.zeros(win_shape, dtype=np.uint8)
        if boundary_mask:
            bound_mask = mask.copy()
        if distance_mask:
            dist_mask = mask.copy()
        return mask, bound_mask, dist_mask

    mask = rasterize(
        polygons,
        win_shape,
        default_value=255,
        transform=transform,
    )
    if boundary_mask or distance_mask or extent_no_border:
        lines = _get_linestrings_from_polygons(polygons)
        bound_mask = rasterize(lines, win_shape, default_value=255, transform=transform)
        if extent_no_border or distance_mask:
            mask_no_bounds = mask.copy()
            mask_no_bounds[bound_mask != 0] = 0
            if extent_no_border:
                mask = mask_no_bounds
            if distance_mask:
                dist_mask = cv2.distanceTransform(
                    mask_no_bounds, cv2.DIST_L2, 3
                ).astype(np.uint8)
    return mask, bound_mask, dist_mask


def _get_linestrings_from_polygons(polys):
    for poly in polys:
        boundary = poly.boundary
        if boundary.type == "MultiLineString":
            for line in boundary.geoms:
                yield line
        else:
            yield boundary


def prepare_label_shapes(
    labels, mask_type="class", label_property="class", classes=None
):
    if mask_type in ("class", "single"):
        polys_dict = classify_polygons(labels, label_property, classes)
        return polys_dict
    else:
        raise RuntimeError(f"mask type '{mask_type}' not supported")


def classify_polygons(labels, label_property, classes):
    with fiona.open(labels) as blocks:
        _logger.info("Found %d labels on label file %s", len(blocks), labels)
        if classes is not None and label_property:
            if label_property not in blocks.schema["properties"]:
                raise ValueError(
                    f"label property '{label_property}' not present in labels schema"
                )
        class_counts = defaultdict(int)
        missing_classes = set()
        polys = defaultdict(list)
        for block in blocks:
            if classes is None:
                c = "_any"
            else:
                c = str(block["properties"][label_property])
                class_counts[c] += 1
                if c not in classes:
                    missing_classes.add(c)
                    continue
            try:
                geom = shape(block["geometry"])
            except RuntimeError:
                _logger.warning(
                    "Failed to get geometry shape for feature: %s", block
                )
                continue
            polys[c].append(geom)
    if missing_classes:
        _logger.warn(
            f"Some features with classes {missing_classes} were ignored as they are not in the specified classes list"
        )
        for c in missing_classes:
            _logger.warn(f"Num. features of class '{c}': {class_counts[c]}")
    return polys
