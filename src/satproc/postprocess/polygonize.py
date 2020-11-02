import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
from functools import partial
from glob import glob

from satproc.utils import grouper, map_with_threads, run_command
from tqdm import tqdm


def gdal_polygonize(src, dst):
    run_command(f'gdal_polygonize.py {src} {dst}')


def process_image(image, *, tmpdir):
    src = image
    name, _ = os.path.splitext(os.path.basename(image))
    dst = os.path.join(tmpdir, f'{name}.gpkg')
    gdal_polygonize(src, dst)


def merge_vector_files(*, input_dir, output, tmpdir):
    srcs = list(glob(os.path.join(input_dir, '*.gpkg')))
    src_groups = list(enumerate(grouper(srcs, n=1000)))
    groups_dir = os.path.join(tmpdir, 'groups')
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
    output_dir = os.path.join(tmpdir, 'temp')
    worker = partial(merge_chip_vector_files, output_dir=output_dir)
    map_with_threads(src_groups, worker)

    # Second, merge ogrmerge results using ogr2ogr into a single file
    group_paths = glob(os.path.join(groups_dir, '*.gpkg'))
    for src in tqdm(group_paths):
        run_command(f'ogr2ogr -f GPKG -update -append {output} {src}',
                    quiet=False)


def polygonize(threshold=None, temp_dir=None, *, input_dir, output):
    images = list(glob(os.path.join(input_dir, '*.tif')))

    must_remove_temp_dir = False
    if temp_dir:
        # Make sure directory exists
        os.makedirs(temp_dir, exist_ok=True)
    else:
        # Create a temporary directory if user did not provide one
        must_remove_temp_dir = True
        tmpdir = tempfile.TemporaryDirectory()
        temp_dir = tmpdir.name

    # Process all chip images
    worker = partial(process_image, tmpdir=temp_dir, threshold=threshold)
    map_with_threads(images, worker)

    # Merge all vector files into a single one
    merge_vector_files(input_dir=temp_dir, output=output, tmpdir=temp_dir)

    if must_remove_temp_dir:
        tmpdir.cleanup()
