# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* New sliding window modes: `exact`, `whole`, and `whole_overlap`
* Add `--skip-low-contrast` to skip low contrast images on `extract_chips`
* New `--masks` option to generate `extent`, `boundary` and `distance` masks on `extract_chips`
* New `satproc_make_masks` console script for only generating masks of images

### Changed

* Use `whole_overlap` when extracting chips by default
* Rename `--write-geojson` option to `--write-footprints`
* Do not skip low contrast images by default
* Bugfix when using `--rescale` with values mode
