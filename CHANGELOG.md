# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `extract_chips`, `make_masks`: Add `--skip-with-empty-mask` and
  `--no-skip-with-empty-mask` options to skip chips with empty masks.
- `smooth_stitch`: Add `--power`/`-p` option to specify spline exponent
- `smooth_stitch`: Add `--temp-dir` to specify temporary directory
- New `spatial_filter` script to apply image filters (median filter and
  gaussian blur) (#43)
- `extract_chips`, `make_masks`: Add `--mask-type single` option (#44)

### Changed

- Do not write image and masks if all masks are empty by default (#29)
- Try to improve smooth stitching process by doing a max-based merge and some
  other fixes (#42)

### Fixed

- If `--step-size` is not specified, it should use the same `--size` (no
  overlapping) (#45)
- Update `rasterio` to 1.3b1 to fix issue with Python 3.10 (#47)

## [0.1.9] - 2022-01-12

### Changed

- Lowered Python requirement to 3.7+

## [0.1.8] - 2022-12-28

### Added

- Define `satproc_make_masks` console script for only generating masks of
  images (#15)
- Define `satproc_generalize` console script for simplifying and smoothing
  polygons (#22)
- Add new sliding window modes: `exact`, `whole`, and `whole_overlap`
- Add `--skip-low-contrast` to skip low contrast images on `extract_chips`
  (#10)
- Add `--masks` option to generate `extent`, `boundary` and `distance` masks on
  `extract_chips`
- Add `--extent-no-border` option on `make_masks` and `extract_chips` for
  removing polygon boundary from extent mask. (#21)

### Changed

- Use `whole_overlap` when extracting chips by default
- Dissolve adjacent polygons from windows after merging groups (#19)
- Rename `--write-geojson` option to `--write-footprints`
- Do not skip low contrast images by default (#10)
- Bugfix when using `--rescale` with values mode
- Use tqdm in ASCII mode to be friendlier on log files
- Pin pyproj>3 version

### Removed

- `build_dataset` module and CLI script
