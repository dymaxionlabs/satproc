#!/usr/bin/env python3

import csv
import json
import os
import random

import fiona
import numpy as np
import rasterio
from pycococreatortools import pycococreatortools
from rasterio.features import rasterize
from rasterio.windows import bounds
from shapely.geometry import box, shape

from satproc.chips import write_image
from satproc.utils import rescale_intensity, sliding_windows

output_tiles = "chips/"
output_tiles_gt = "chips_gt/"

CATEGORIES = [{"id": 1, "name": "field", "supercategory": "shape", "is_crowd": True}]

coco_output = {
    "info": {},
    "licenses": [],
    "categories": CATEGORIES,
    "images": [],
    "annotations": [],
}

COCO = "COCO"
RETINANET = "RETINANET"


def mask_from_polygons(polygons, win, mask_path, src, kwargs, image_index, mask_index):
    if polygons:
        mask = rasterize(
            polygons,
            (win.height, win.width),
            transform=rasterio.windows.transform(win, src.transform),
        )
        if mask is None:
            Exception(
                "A empty mask Was generated - Image {} Mask {}".format(
                    image_index, mask_index
                )
            )
    else:
        mask = np.zeros((win.height, win.width), dtype=np.uint8)

    # Write tile
    kwargs.update(dtype=rasterio.uint8, count=1, nodata=0)
    dst_name = "{}/{}_{}.tif".format(mask_path, image_index, mask_index)
    with rasterio.open(dst_name, "w", **kwargs) as dst:
        dst.write(mask, 1)

    return mask


def constrain_and_scale(coord, max_value, IMAGE_TILE_SIZE):
    return round((min(max(coord, 0), IMAGE_TILE_SIZE) / IMAGE_TILE_SIZE) * max_value)


def train_val_split_rows(rows, val_size=0.2):
    random.shuffle(rows)
    n_val_size = round(len(rows) * val_size)
    return rows[n_val_size:], rows[:n_val_size]


def build_dataset(
    dataset_path,
    rasters,
    chip_size=512,
    step_size=128,
    output_dir=".",
    instance=True,
    type=COCO,
    label="unknown",
    rescale_mode=None,
    rescale_range=None,
    bands=[1, 2, 3],
):

    blocks = fiona.open(dataset_path)
    block_shapes = [shape(b["geometry"]) for b in blocks]
    type = type.upper()

    k = 0
    a = 0

    if type == COCO:
        tile_path = os.path.join(output_dir, output_tiles)
        tile_gt_path = os.path.join(output_dir, output_tiles_gt)
        os.makedirs(tile_gt_path, exist_ok=True)
        os.makedirs(tile_path, exist_ok=True)

    if type == RETINANET:
        rows = []

    for raster in rasters:
        k = 0
        with rasterio.open(raster) as src:
            tile_path = os.path.join(output_dir, output_tiles, os.path.basename(raster))
            os.makedirs(tile_path, exist_ok=True)
            win_size = (chip_size, chip_size)
            win_step_size = (step_size, step_size)
            windows = list(
                sliding_windows(
                    win_size, win_step_size, src.shape[1], src.shape[0], whole=True
                )
            )

            for (win, (i, j)) in windows:
                if type == COCO:
                    # Write tif tile
                    kwargs = src.meta.copy()
                    kwargs.update(
                        {
                            "height": win.height,
                            "width": win.width,
                            "transform": rasterio.windows.transform(win, src.transform),
                        }
                    )
                    dst_name = "{}/{}.tif".format(tile_path, k)
                    with rasterio.open(dst_name, "w", **kwargs) as dst:
                        dst.write(src.read(window=win))

                    # Append tile info on COCO dataset

                    image_info = pycococreatortools.create_image_info(
                        k, os.path.basename(dst_name), (win.height, win.width)
                    )
                    coco_output["images"].append(image_info)

                    # Get intersecting shapes with current window
                    bbox_shape = box(*bounds(win, src.transform))
                    intersect_polys = [
                        bbox_shape.intersection(s)
                        for s in block_shapes
                        if bbox_shape.intersects(s)
                    ]

                    if len(intersect_polys) > 0:
                        if instance:
                            # For each polygon, create a mask
                            for i, poly in enumerate(intersect_polys):
                                mask = mask_from_polygons(
                                    [poly], win, tile_gt_path, src, kwargs, k, i
                                )

                                if mask is not None and type == COCO:
                                    # Append annotation info on COCO dataset
                                    annotation_info = (
                                        pycococreatortools.create_annotation_info(
                                            a,
                                            k,
                                            CATEGORIES[0],
                                            mask,
                                            (512, 512),
                                            tolerance=2,
                                        )
                                    )
                                    a = a + 1
                                    coco_output["annotations"].append(annotation_info)
                        else:
                            mask = mask_from_polygons(
                                intersect_polys, win, tile_gt_path, src, kwargs, k, 0
                            )

                            if mask is not None and type == COCO:
                                # Append annotation info on COCO dataset
                                annotation_info = (
                                    pycococreatortools.create_annotation_info(
                                        a,
                                        k,
                                        CATEGORIES[0],
                                        mask,
                                        (512, 512),
                                        tolerance=2,
                                    )
                                )
                                a = a + 1
                                coco_output["annotations"].append(annotation_info)
                    else:
                        # If there aren't any polygon, an empty mask is saved
                        mask = mask_from_polygons(
                            None, win, tile_gt_path, src, kwargs, k, i
                        )
                elif type == RETINANET:
                    img = src.read(window=win)
                    img = np.nan_to_num(img)
                    img = np.array([img[b - 1, :, :] for b in bands])

                    if rescale_mode:
                        img = rescale_intensity(img, rescale_mode, rescale_range)

                    dst_name = "{}/{}.jpg".format(tile_path, k)
                    write_image(img, dst_name)

                    # Generate segments
                    segments = []
                    win_bounds = rasterio.windows.bounds(win, src.transform)
                    window_box = box(*win_bounds)

                    hits = [
                        hit
                        for _, hit in blocks.items(
                            bbox=(
                                win_bounds[0],
                                win_bounds[1],
                                win_bounds[2],
                                win_bounds[3],
                            )
                        )
                    ]
                    for hit in hits:
                        hit_shape = shape(hit["geometry"])
                        bbox = box(*hit_shape.bounds)

                        inter_bbox = window_box.intersection(bbox)
                        inter_bbox_bounds = inter_bbox.bounds
                        win_transform = rasterio.windows.transform(win, src.transform)
                        minx, maxy = ~win_transform * (
                            inter_bbox_bounds[0],
                            inter_bbox_bounds[1],
                        )
                        maxx, miny = ~win_transform * (
                            inter_bbox_bounds[2],
                            inter_bbox_bounds[3],
                        )
                        segment = dict(
                            x=minx,  # - index[0],
                            y=miny,  # - index[1],
                            width=round(maxx - minx),
                            height=round(maxy - miny),
                            label=label,
                        )
                        if segment["width"] > 0 and segment["height"] > 0:
                            segments.append(segment)

                    # Generate CSVs
                    w = chip_size
                    h = chip_size
                    for s in segments:
                        row = {}
                        x1, x2 = sorted([s["x"], s["x"] + s["width"]])
                        y1, y2 = sorted([s["y"], s["y"] + s["height"]])
                        row["x1"] = constrain_and_scale(x1, w, chip_size)
                        row["x2"] = constrain_and_scale(x2, w, chip_size)
                        row["y1"] = constrain_and_scale(y1, h, chip_size)
                        row["y2"] = constrain_and_scale(y2, h, chip_size)
                        row["tile_path"] = "{}.jpg".format(
                            os.path.join(output_tiles, os.path.basename(raster), str(k))
                        )
                        row["label"] = s["label"]
                        rows.append(row)
                else:
                    pass
                k = k + 1

    if type == COCO:
        with open(os.path.join(output_dir, "annotations_coco.json"), "w") as outfile:
            json.dump(coco_output, outfile)

    if type == RETINANET:
        rows_train, rows_val = train_val_split_rows(rows)
        for name, rows in zip(["train", "val"], [rows_train, rows_val]):
            file_path = os.path.join(output_dir, "{}.csv".format(name))
            with open(file_path, "w") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=("tile_path", "x1", "y1", "x2", "y2", "label")
                )
                for row in rows:
                    writer.writerow(row)
        with open(os.path.join(output_dir, "classes.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=("label", "index"))
            writer.writerow({"label": label, "index": 0})
