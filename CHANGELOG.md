# Changelog

## 0.6.0 (Unreleased)

Introduce `TableCell` dataclass, reorganize grid conversion code, and add HTML-to-grid convenience functions.

### Added
- `TableCell` dataclass with `is_row_header` field as a typed replacement for cell dicts.
- `html_to_grids` convenience function for converting HTML tables directly to content and topology grids.

### Changed
- Renamed `utils.py` to `conversion.py` and moved grid conversion functions out of `grits.py` into it.
- Renamed `cell_dicts` to `cell_list` throughout to reflect the shift to `TableCell` objects.

## 0.5.0 (Unreleased)

Major evaluator overhaul: simplify the interface, improve naming, add convenience functions, and switch to `pylcs`.

### Added
- `hungarian_grits_top`, `hungarian_grits_con`, and `hungarian_grits_loc` convenience functions.
- `compute_counts()` method on the evaluator.
- HTML integration tests with tightened assertions to exact values.

### Changed
- Renamed `GritsResult` to `GritsMatchingResult` and `HungarianGritsResult` to `HungarianGritsMatchingResult`.
- Renamed `compute_avg_grits` to `compute_avg_grits_per_sample`; output keys now use `avg_grits_METRIC_per_sample`.
- Renamed `avg` to `mean` in evaluator output.
- `eval_htmls` and `eval_table_cell_dicts` now evaluate only the configured metrics instead of all three.
- Switched LCS implementation to `pylcs` for performance.

## 0.4.3 (Unreleased)

Split the evaluator's `eval_sample` and refine the evaluator API.

### Changed
- Split `GritsEvaluator.eval_sample` into `eval_grids` and `eval_grids_by_metric`.
- Renamed `eval_html` to `eval_htmls`.
- Renamed `_GritsMetricEvaluator.eval_sample` to `eval_grids` and added input validation.

## 0.4.2 (Unreleased)

Consolidate to a single `GritsEvaluator` interface and require list inputs.

### Changed
- Consolidated `MultiGritsEvaluator` into a single `GritsEvaluator` that can track multiple metrics.
- `GritsEvaluator.eval_sample` now accepts lists directly when tracking a single metric.
- `eval_html` now requires list inputs.

## 0.4.1 (Unreleased)

Add convenience scoring functions and rename matching functions for clarity.

### Added
- `grits_con`, `grits_top`, `grits_loc` convenience functions that return F-score tuples.

### Changed
- Renamed `hungarian_grits_con/top/loc` to `hungarian_grits_con/top/loc_matching`.
- Renamed various internal matching functions for consistency.

## 0.4.0 (Unreleased)

Major evaluator improvements: structured result types, exact match tracking, multi-metric evaluation, and API cleanup.

### Added
- `GritsResult` dataclass for `grits_con/top/loc` return values.
- `HungarianGritsResult` dataclass replacing raw tuples from `hungarian_grits`.
- `MultiGritsEvaluator` for evaluating all three metrics simultaneously.
- `eval_table_cells()` for evaluating all three metrics from cell dicts.
- Exact match tracking: `is_exact_match` on results, `exact_match_accuracy` and `avg_exact_match_accuracy` on the evaluator.
- Grid count tracking (`num_true_grids`, `num_pred_grids`) in the evaluator.
- `eval_html()` now accepts lists of HTMLs.

### Changed
- Metrics returned as dict instead of custom object.
- `GritsResult.is_exact_match` uses `bool` instead of `int`.
- Renamed `cells_to_grid_con/loc/top` to `cell_dicts_to_grid_con/loc/top`.
- Renamed `cells` parameters to `cell_dicts` throughout for clarity.
- Simplified `eval_table_cells()` signature to accept only list of tables.
- Require matching true and pred cell counts for exact match.

### Removed
- `append_result()` method on the evaluator (in favor of simpler API).

## 0.3.1 (Unreleased)

Renaming variables and simplifying functions to make the default behavior clearer.

### Changed
- Always use default value for missing cells to make actual behavior clearer.
- Improved function naming and refactored tests.

## 0.3.0 (Unreleased)

Rename conversion functions in grits.py for consistency and clarity.

### Changed
- Refactored `cells_to_grid` into two separately callable functions for content and location.
- Renamed `cells_to_relspan_grid` for clarity.

## 0.2.0 (Unreleased)

Rename core functions in GritsEvaluator for clarity and add support for GriTS-Loc.

### Added
- GriTS-Loc metric: `hungarian_grits_loc` and integration in `GritsEvaluator`.

### Changed
- Renamed evaluator functions for clarity and consistency.
- Always use default value for missing cells to make actual behavior clearer.
- Refactored tests.

## 0.1.1 (Unreleased)

Update project metadata for test release.

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
