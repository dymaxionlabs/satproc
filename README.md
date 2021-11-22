# satproc

Python library and CLI tools for processing geospatial imagery for ML

## Description

satproc helps you work with large amount of geospatial raster images
(satellite, drone, etc.) and process them for training machine learning, for
object detection or semantic segmentation problems.

## Installation

To install the latest stable version run:

```
pip install pysatproc
```

You can also clone the repository at https://github.com/dymaxionlabs/satproc/
and install the development version with the `-e` option, like this:

```
git clone https://github.com/dymaxionlabs/satproc.git
pip install -e satproc/
```

Now, whenever you want to bring the latest changes, just run `git pull` from the
cloned repository.

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

## Contributing

Bug reports and pull requests are welcome on GitHub at the [issues
page](https://github.com/dymaxionlabs/satproc). This project is intended to be
a safe, welcoming space for collaboration, and contributors are expected to
adhere to the [Contributor Covenant](http://contributor-covenant.org) code of
conduct.

## License

This project is licensed under Apache 2.0. Refer to [LICENSE.txt](./LICENSE.txt).
