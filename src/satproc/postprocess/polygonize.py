import logging
import os
import tempfile
from functools import partial
from glob import glob

import fiona
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from satproc.utils import grouper, map_with_threads, run_command

_logger = logging.getLogger(__name__)


def gdal_polygonize(src, dst):
    run_command(f"gdal_polygonize.py {src} {dst}")


def apply_threshold(src, dst, value=None, *, threshold):
    """
    Output source values (probabilities) instead of simply a binary mask

    Make sure nodata=0, so that gdal_polygonize step ignores pixels under
    threshold.

    Parameters
    ----------
    src : str
        path to input raster
    dst : str
        path to output raster
    threshold : int
        threshold value
    value : int
        value to use on raster output when values are over threshold
        if None, use the original value from src

    """
    # Rescale to 256
    threshold = threshold * 256

    if not value:
        value = "A"
    run_command(
        "gdal_calc.py "
        f'--calc "(A >= {threshold}) * {value}" '
        f"-A {src} "
        "--NoDataValue 0 "
        f"--outfile {dst}"
    )


def process_image(image, value=None, *, tmpdir, threshold):
    src = image
    if threshold:
        src = os.path.join(tmpdir, os.path.basename(image))
        apply_threshold(src=image, dst=src, threshold=threshold, value=value)
    name, _ = os.path.splitext(os.path.basename(image))
    dst = os.path.join(tmpdir, f"{name}.gpkg")
    gdal_polygonize(src, dst)


def merge_vector_files(*, input_dir, output, tmpdir):
    srcs = list(glob(os.path.join(input_dir, "*.gpkg")))
    src_groups = list(enumerate(grouper(srcs, n=1000)))
    groups_dir = os.path.join(tmpdir, "groups")
    os.makedirs(groups_dir, exist_ok=True)

    def merge_chip_vector_files(enumerated_srcs, *, output_dir):
        i, srcs = enumerated_srcs
        srcs = [f for f in srcs if f]
        output = os.path.join(groups_dir, f"{i}.gpkg")
        run_command(
            f"ogrmerge.py -overwrite_ds -single "
            f'-f GPKG -o {output} {" ".join(srcs)}',
            quiet=False,
        )
        return output

    # First, merge groups of vector files using ogrmerge.py in parallel
    output_dir = os.path.join(tmpdir, "temp")
    worker = partial(merge_chip_vector_files, output_dir=output_dir)
    map_with_threads(src_groups, worker, desc="Merge chips into groups")

    # Second, merge ogrmerge results using ogr2ogr into a single file
    group_paths = glob(os.path.join(groups_dir, "*.gpkg"))
    merged_path = os.path.join(output_dir, "merged", os.path.basename(output))
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    with logging_redirect_tqdm():
        for src in tqdm(group_paths, ascii=True, desc="Merge groups"):
            run_command(
                f"ogr2ogr -f GPKG -update -append {merged_path} {src}", quiet=False
            )

    # Third, dissolve the final vector file
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with fiona.open(merged_path) as src:
        if len(src) == 0:
            _logger.warn("No shapes. Will not write output file")
            return
        with fiona.open(output, "w", **src.meta) as dst:
            all_shapes = list(
                grouper(
                    (
                        shape(feat["geometry"]).buffer(0)
                        for feat in tqdm(
                            src, total=len(src), ascii=True, desc="Reading shapes"
                        )
                    ),
                    n=10000,
                )
            )

            single_shapes = []
            for group_shapes in tqdm(all_shapes, ascii=True, desc="Dissolve"):
                group_shapes = [s for s in group_shapes if s]
                single_shape = unary_union(group_shapes)
                single_shapes.append(single_shape)

            _logger.info("Dissolve final shapes")
            final_shape = unary_union(single_shapes)
            props = {"DN": 255}
            for s in _multipart_to_single_parts(final_shape):
                dst.write({"geometry": mapping(s), "properties": props})
    _logger.info("%s written", output)


def _multipart_to_single_parts(shp):
    if shp.type == "MultiPolygon":
        for s in shp.geoms:
            yield s
    elif shp.type == "Polygon":
        yield shp
    else:
        raise RuntimeError(
            f"shape should be a Polygon or MultiPolygon but was {shp.type}"
        )


def retile_all(input_files, tile_size, temp_dir):
    tiles_dir = os.path.join(temp_dir, "_tiles")
    for raster in input_files:
        retile(raster, output_dir=tiles_dir, tile_size=tile_size)
    tiles_files = list(glob(os.path.join(tiles_dir, "*.tif")))
    _logger.info(
        "Num. tiles sized %dx%d at %s: %d",
        tile_size,
        tile_size,
        tiles_dir,
        len(tiles_files),
    )
    return tiles_files


def retile(raster, output_dir, tile_size):
    os.makedirs(output_dir, exist_ok=True)
    run_command(
        "gdal_retile.py "
        f"-ps {tile_size} {tile_size} "
        f"-targetDir {output_dir} {raster}"
    )


def polygonize(
    threshold=None,
    value=None,
    temp_dir=None,
    input_files=[],
    input_dir=None,
    tile_size=None,
    *,
    output,
):
    if input_dir:
        input_files = list(glob(os.path.join(input_dir, "*.tif")))

    if not input_files:
        raise ValueError("No input files")

    must_remove_temp_dir = False
    if temp_dir:
        # Make sure directory exists
        os.makedirs(temp_dir, exist_ok=True)
    else:
        # Create a temporary directory if user did not provide one
        must_remove_temp_dir = True
        tmpdir = tempfile.TemporaryDirectory()
        temp_dir = tmpdir.name

    # Retile input images before processing (if tile_size is present)
    if tile_size:
        input_files = retile_all(input_files, tile_size, temp_dir)

    # Process all chip images
    worker = partial(process_image, tmpdir=temp_dir, threshold=threshold, value=value)
    map_with_threads(input_files, worker, desc="Polygonize chips")

    # Merge all vector files into a single one
    merge_vector_files(input_dir=temp_dir, output=output, tmpdir=temp_dir)

    if must_remove_temp_dir:
        tmpdir.cleanup()
