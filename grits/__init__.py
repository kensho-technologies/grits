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
    cells_to_grid_con,
    cells_to_grid_loc,
    cells_to_grid_top,
    compute_fscore,
    grits_con,
    grits_loc,
    grits_top,
    hungarian_grits_con,
    hungarian_grits_loc,
    hungarian_grits_top,
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
    "cells_to_grid_con",
    "cells_to_grid_loc",
    "cells_to_grid_top",
    "html_to_cell_dicts",
    # GriTS metrics
    "GritsResult",
    "HungarianGritsResult",
    "grits_con",
    "grits_loc",
    "grits_top",
    "hungarian_grits_con",
    "hungarian_grits_loc",
    "hungarian_grits_top",
    "GritsEvaluator",
    "MultiGritsEvaluator",
    # utilities
    "compute_fscore",
    "iou",
    "lcs_similarity",
]
