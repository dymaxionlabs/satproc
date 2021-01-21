import logging
import os
import tempfile
from functools import partial
from glob import glob

from satproc.utils import grouper, map_with_threads, run_command
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def gdal_polygonize(src, dst):
    run_command(f'gdal_polygonize.py {src} {dst}')


def process_image(image, *, temp_dir):
    src = image
    name, _ = os.path.splitext(os.path.basename(image))
    dst = os.path.join(temp_dir, f'{name}.gpkg')
    gdal_polygonize(src, dst)


def merge_vector_files(*, input_dir, output, temp_dir):
    srcs = list(glob(os.path.join(input_dir, '*.gpkg')))
    src_groups = list(enumerate(grouper(srcs, n=1000)))
    groups_dir = os.path.join(temp_dir, '_groups')
    os.makedirs(groups_dir, exist_ok=True)

    def merge_chip_vector_files(enumerated_srcs, *, output_dir):
        i, srcs = enumerated_srcs
        srcs = [f for f in srcs if f]
        output = os.path.join(groups_dir, f'{i}.gpkg')
        run_command(
            f'ogrmerge.py -overwrite_ds -single '
            f'-f GPKG -o {output} {" ".join(srcs)}',
            quiet=False)
        return output

    # First, merge groups of vector files using ogrmerge.py in parallel
    output_dir = os.path.join(temp_dir, '_merge')
    worker = partial(merge_chip_vector_files, output_dir=output_dir)
    map_with_threads(src_groups, worker)

    # Second, merge ogrmerge results using ogr2ogr into a single file
    group_paths = glob(os.path.join(groups_dir, '*.gpkg'))
    for src in tqdm(group_paths):
        run_command(f'ogr2ogr -f GPKG -update -append {output} {src}',
                    quiet=False)


def retile_all(input_files, tile_size, temp_dir):
    tiles_dir = os.path.join(temp_dir, '_tiles')
    for raster in input_files:
        retile(raster, output_dir=tiles_dir, tile_size=tile_size)
    tiles_files = list(glob(os.path.join(tiles_dir, '*.tif')))
    _logger.info("Num. tiles sized %dx%d at %s: %d", tile_size, tile_size,
                 tiles_dir, len(tiles_files))
    return tiles_files


def retile(raster, output_dir, tile_size):
    os.makedirs(output_dir, exist_ok=True)
    run_command('gdal_retile.py ' \
        f'-ps {tile_size} {tile_size} ' \
        f'-targetDir {output_dir} {raster}')


def polygonize(temp_dir=None, tile_size=None, *, input_files, output):
    tmpdir = None
    if temp_dir:
        # Make sure directory exists
        os.makedirs(temp_dir, exist_ok=True)
    else:
        # Create a temporary directory if user did not provide one
        tmpdir = tempfile.TemporaryDirectory()
        temp_dir = tmpdir.name

    tile_files = input_files
    if tile_size:
        # If tile_size was provided, make sure we deal with only rasters of size
        # no larger than `tile_size`, by *retiling* all input rasters.
        tile_files = retile_all(input_files,
                                tile_size=tile_size,
                                temp_dir=temp_dir)

    # Process all tile images
    worker = partial(process_image, temp_dir=temp_dir)
    map_with_threads(tile_files, worker)

    # Merge all vector files into a single one
    merge_vector_files(input_dir=temp_dir, output=output, temp_dir=temp_dir)

    if tmpdir:
        tmpdir.cleanup()
