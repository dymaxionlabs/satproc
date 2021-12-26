# Getting started

**satproc** helps you work with large amount of geospatial raster images
(satellite, drone, etc.) and process them for training machine learning, for
object detection or semantic segmentation problems.

You can either use it from the command line or from Python as a library.

## Usage

### Command Line (CLI)

When installed, satproc makes available a series of command-line scripts to
process files without resorting to writing a Python script.

* `satproc_extract_chips`: Extract chips from raster images, optionally creating
  masks for each chip using a labels vector file.
* `satproc_make_masks`: Create masks from raster images and a labels vector file.
* `satproc_polygonize`: Polygonizes chip images into a single polygon vector file.
* `satproc_generalize`: Generalizes vector files by simplyfing and smoothing polygon boundary lines.
* `satproc_smooth_stitch`: Smoothes overlapping probability result chips.
* `satproc_scale`: Rescales values from raster images
* `satproc_match_histograms`: Matches histograms of raster images from a reference image.

Run any command with the `-h`/`--help` flag to see the available options and
information on how to use them.