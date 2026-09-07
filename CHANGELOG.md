# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]


## [1.0.4] - 2026-09-07

### Fixed
- fix: Process PDF files correctly
- Task progress now advances per input file and per PDF page instead of being
  pinned at 15%.
- Downloaded input files are removed once processed instead of accumulating in
  the temporary directory.
- perf: Reduce DPI and use_cache=True to speedup processing
- fix: Increase MAX_OUTPUT_TOKENS

## [1.0.3] - 2026-07-27

### Fixed
- Fix build

## [1.0.2] - 2026-07-27

### Fixed
- Fix build

## [1.0.1] - 2026-07-27

### Fixed
- Corrected minor bugs in the OCR processing module.

## [1.0.0] - 2025-12-08

Initial release