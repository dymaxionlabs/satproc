# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.9

### Changes

* Lowered Python requirement to 3.7+

## 0.1.8

### Added

* Define `satproc_make_masks` console script for only generating masks of
  images (#15)
* Define `satproc_generalize` console script for simplifying and smoothing
  polygons (#22)
* Add new sliding window modes: `exact`, `whole`, and `whole_overlap`
* Add `--skip-low-contrast` to skip low contrast images on `extract_chips`
  (#10)
* Add `--masks` option to generate `extent`, `boundary` and `distance` masks on
  `extract_chips`
* Add `--extent-no-border` option on `make_masks` and `extract_chips` for
  removing polygon boundary from extent mask. (#21)

### Changed

* Use `whole_overlap` when extracting chips by default
* Dissolve adjacent polygons from windows after merging groups (#19)
* Rename `--write-geojson` option to `--write-footprints`
* Do not skip low contrast images by default (#10)
* Bugfix when using `--rescale` with values mode
* Use tqdm in ASCII mode to be friendlier on log files
* Pin pyproj>3 version

### Removed

* `build_dataset` module and CLI script
