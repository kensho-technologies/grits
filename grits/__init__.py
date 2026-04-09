# Copyright 2025-present Kensho Technologies, LLC.
from .bbox import (
    BBox,
    BBoxProtocol,
    get_horizontal_overlap,
    get_overlap_area,
    get_vertical_overlap,
    is_horizontal_overlap,
    is_vertical_overlap,
)
from .grits import (
    GritsEvaluator,
    GritsResult,
    HungarianGritsResult,
    MultiGritsEvaluator,
    cell_dicts_to_grid_con,
    cell_dicts_to_grid_loc,
    cell_dicts_to_grid_top,
    compute_fscore,
    grits_con,
    grits_con_matching,
    grits_loc,
    grits_loc_matching,
    grits_top,
    grits_top_matching,
    hungarian_grits_con_matching,
    hungarian_grits_loc_matching,
    hungarian_grits_top_matching,
    iou,
    lcs_similarity,
)
from .utils import html_to_cell_dicts


__all__ = [
    # bbox types and utilities
    "BBox",
    "BBoxProtocol",
    "get_horizontal_overlap",
    "get_overlap_area",
    "get_vertical_overlap",
    "is_horizontal_overlap",
    "is_vertical_overlap",
    # grid conversion
    "cell_dicts_to_grid_con",
    "cell_dicts_to_grid_loc",
    "cell_dicts_to_grid_top",
    "html_to_cell_dicts",
    # GriTS metrics
    "GritsResult",
    "HungarianGritsResult",
    "grits_con",
    "grits_con_matching",
    "grits_loc",
    "grits_loc_matching",
    "grits_top",
    "grits_top_matching",
    "hungarian_grits_con_matching",
    "hungarian_grits_loc_matching",
    "hungarian_grits_top_matching",
    "GritsEvaluator",
    "MultiGritsEvaluator",
    # utilities
    "compute_fscore",
    "iou",
    "lcs_similarity",
]
