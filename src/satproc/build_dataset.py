#!/usr/bin/env python3

from pycococreatortools import pycococreatortools
from rasterio.features import rasterize
from rasterio.windows import bounds, Window
from shapely.geometry import Polygon, shape, box
from shapely.ops import unary_union
import datetime
import fiona
import json
import numpy as np
import os
import rasterio

output_tiles = 'chips/'
output_tiles_gt = 'chips_gt/'

CATEGORIES = [{
    'id': 1,
    'name': 'field',
    'supercategory': 'shape',
    'is_crowd': True
}]

coco_output = {
    "info": {},
    "licenses": [],
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}


def sliding_windows(size, width, height):
    """Slide a window of +size+ pixels"""
    for i in range(0, height, size):
        for j in range(0, width, size):
            yield Window(j, i, min(width - j, size), min(height - i, size))



def mask_from_polygons(polygons, win, mask_path, image_index, mask_index):
    if polygons:
        mask = rasterize(polygons, (win.height, win.width),
            transform=rasterio.windows.transform(win, src.transform))
    else: 
        mask = np.zeros((win.height, win.width), dtype=np.uint8)

    # Write tile
    kwargs.update(dtype=rasterio.uint8, count=1, nodata=0)
    dst_name = '{}/{}_{}.tif'.format(mask_path, image_index, mask_index)
    with rasterio.open(dst_name, 'w', **kwargs) as dst:
        dst.write(mask, 1)

    return mask


def build_dataset(dataset_path, raster, chip_size = 512, output_dir=".", instance = True, coco_output = False):

    blocks = fiona.open(dataset_path)
    block_shapes = [shape(b['geometry']) for b in blocks]

    k = 0
    a = 0
    os.makedirs(os.path.join(output_dir,output_tiles), exist_ok=True)
    os.makedirs(os.path.join(output_dir,output_tiles_gt), exist_ok=True)

    with rasterio.open(raster) as src:
        windows = list(sliding_windows(chip_size, src.shape[1], src.shape[0]))

        for win in windows:
            # Write tif tile
            kwargs = src.meta.copy()
            kwargs.update({
                'height': win.height,
                'width': win.width,
                'transform': rasterio.windows.transform(win, src.transform)
            })
            dst_name = '{}/{}.tif'.format(os.path.join(output_dir,output_tiles), k)
            with rasterio.open(dst_name, 'w', **kwargs) as dst:
                dst.write(src.read(window=win))

            # Append tile info on COCO dataset
            if coco_output:
                image_info = pycococreatortools.create_image_info(
                    k, os.path.basename(dst_name), (win.height, win.width))
                coco_output["images"].append(image_info)

            # Get intersecting shapes with current window
            bbox_shape = box(*bounds(win, src.transform))
            intersect_polys = [
                bbox_shape.intersection(s) for s in block_shapes
                if bbox_shape.intersects(s)
            ]

            if len(intersect_polys) > 0:
                if instance:
                    # For each polygon, create a mask
                    for i, poly in enumerate(intersect_polys):
                        mask = mask_from_polygons(
                            [poly], win, os.path.join(output_dir,output_tiles_gt), k, i)

                        if mask and coco_output:
                            #Append annotation info on COCO dataset
                            annotation_info = pycococreatortools.create_annotation_info(a, k, CATEGORIES[0], mask, (512,512), tolerance=2)
                            a = a + 1
                            coco_output["annotations"].append(annotation_info)
                else:
                    mask = mask_from_polygons(intersect_polys, win, os.path.join(output_dir,output_tiles_gt), k, 0)

                    if mask and coco_output:    
                        #Append annotation info on COCO dataset
                        annotation_info = pycococreatortools.create_annotation_info(a, k, CATEGORIES[0], mask, (512,512), tolerance=2)
                        a = a + 1
                        coco_output["annotations"].append(annotation_info)
            else:
                mask = mask_from_polygons(None, win, os.path.join(output_dir,output_tiles_gt), k, i)

            k = k + 1
    
    with open(os.path.join(output_dir,'annotations_coco.json'), 'w') as outfile:
        json.dump(coco_output, outfile)
