# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added
- Support for PDF input. Multi-page PDFs are rasterized and read page by page,
  and the page texts are returned joined as a single document.
- `OCR_PDF_DPI`, `OCR_MAX_PDF_PAGES` and `OCR_MAX_PAGE_PIXELS` environment
  variables to tune PDF rendering resolution, the page limit per document and
  the maximum rasterized page size.

### Fixed
- Task progress now advances per input file and per PDF page instead of being
  pinned at 15%.
- Downloaded input files are removed once processed instead of accumulating in
  the temporary directory.

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