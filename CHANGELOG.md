# Changelog

## 0.1.1 (2026-04-02)

Updated project metadata for release.

## 0.1.0 (Unreleased)

Initial migration of the GriTS (Grid Table Similarity) metrics package.

### Added
- Core GriTS metric functions: `grits_top`, `grits_loc`, `grits_con` for comparing table topology, location, and content.
- `GritsEvaluator` class for accumulating results across multiple tables and computing aggregate (micro) and average (macro) F1 scores.
- Hungarian algorithm-based multi-grid matching via `hungarian_grits_top` and `hungarian_grits_con`.
- Bounding box utilities (`BBox`, `BBoxProtocol`, overlap functions) in `grits.bbox`.
- Grid conversion helpers: `cells_to_grid`, `cells_to_relspan_grid`.
- HTML table parsing via `html_to_cells`.
- Utility functions: `compute_fscore`, `lcs_similarity`, `iou`.
