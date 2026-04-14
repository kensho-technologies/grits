# Copyright 2025-present Kensho Technologies, LLC.
import dataclasses
from difflib import SequenceMatcher
import itertools
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bbox import BBox, get_overlap_area
from .utils import html_to_cell_dicts


def _cell_dicts_to_grid(
    cell_dicts: list[dict[str, Any]], key: str, default_value: Any
) -> list[list[Any]]:
    """Convert from a list of cell dicts to a matrix of grid cell features.

    This matrix representation is the input to GriTS.

    Args:
        cell_dicts: List of cell dicts with 'row_nums', 'column_nums', and feature keys.
        key: The cell feature key to extract ('bbox' for GriTS_Loc, 'cell_text' for GriTS_Con).
        default_value: Value used to initialize grid cells that are not covered by any cell dict.

    Returns:
        A 2D list (grid) of cell features indexed by [row][column].
    """
    if len(cell_dicts) == 0:
        return [[]]
    num_rows = max([max(cell_dict["row_nums"]) for cell_dict in cell_dicts]) + 1
    num_columns = max([max(cell_dict["column_nums"]) for cell_dict in cell_dicts]) + 1
    cell_grid = [[default_value for _ in range(num_columns)] for _ in range(num_rows)]
    for cell_dict in cell_dicts:
        for row_num in cell_dict["row_nums"]:
            for column_num in cell_dict["column_nums"]:
                cell_grid[row_num][column_num] = cell_dict[key]

    return cell_grid


def cell_dicts_to_grid_con(cell_dicts: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert from a list of cell dicts to a matrix of cell text, used for computing GriTS_Con.

    Args:
        cell_dicts: List of cell dicts with 'row_nums', 'column_nums', and 'cell_text'.

    Returns:
        A 2D list (grid) of cell text strings indexed by [row][column].
    """
    default_value = ""
    return _cell_dicts_to_grid(cell_dicts, key="cell_text", default_value=default_value)


def cell_dicts_to_grid_loc(cell_dicts: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert from a list of cell dicts to a matrix of cell bounding boxes, used for computing GriTS_Loc.

    Args:
        cell_dicts: List of cell dicts with 'row_nums', 'column_nums', and 'bbox'.

    Returns:
        A 2D list (grid) of bounding boxes indexed by [row][column].
    """
    default_value = [0.0, 0.0, 0.0, 0.0]
    return _cell_dicts_to_grid(cell_dicts, key="bbox", default_value=default_value)


def cell_dicts_to_grid_top(cell_dicts: list[dict[str, Any]]) -> list[list[Any]]:
    """Convert from a list of cell dicts to the matrix/grid of relative spans (boxes), used for computing GriTS_Top.

    For the cell at grid location (i,j), let a(i,j) be its rowspan,
    let β(i,j) be its colspan, let p(i,j) be the minimum row it occupies,
    and let θ(i,j) be the minimum column it occupies. Its relative span is
    bounding box [θ(i,j)-j, p(i,j)-i, θ(i,j)-j+β(i,j), p(i,j)-i+a(i,j)].

    It gives the size and location of the cell each grid cell belongs to
    relative to the current grid cell location, in grid coordinate units.
    Note that for a non-spanning cell this will always be [0, 0, 1, 1].

    Args:
        cell_dicts: List of cell dicts with 'row_nums' and 'column_nums'.

    Returns:
        A 2D list (grid) of relative span values indexed by [row][column].
    """
    if len(cell_dicts) == 0:
        return [[]]
    num_rows = max([max(cell_dict["row_nums"]) for cell_dict in cell_dicts]) + 1
    num_columns = max([max(cell_dict["column_nums"]) for cell_dict in cell_dicts]) + 1

    # Initialize grid with [0, 0, 1, 1] (non-spanning cells) instead of 0.0
    # This handles cases where HTML tables have missing cells
    default_value = [0, 0, 1, 1]
    cell_grid = [[default_value for _ in range(num_columns)] for _ in range(num_rows)]

    for cell_dict in cell_dicts:
        min_row_num = min(cell_dict["row_nums"])
        min_column_num = min(cell_dict["column_nums"])
        max_row_num = max(cell_dict["row_nums"]) + 1
        max_column_num = max(cell_dict["column_nums"]) + 1
        for row_num in cell_dict["row_nums"]:
            for column_num in cell_dict["column_nums"]:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num,
                ]

    return cell_grid


def compute_fscore(
    num_true_positives: float, num_true: int, num_positives: int
) -> tuple[float, float, float]:
    """Compute the F-score or F1-measure for a collection of predictions.

    Conventions:
    - Precision is 1 when there are no predicted instances
    - Recall is 1 when there are no true instances
    - F-score is 0 when recall or precision is 0

    Args:
        num_true_positives: Number of correctly predicted instances.
        num_true: Total number of true instances.
        num_positives: Total number of predicted instances.

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    if num_positives > 0:
        precision = num_true_positives / num_positives
    else:
        precision = 1.0
    if num_true > 0:
        recall = num_true_positives / num_true
    else:
        recall = 1.0

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return fscore, precision, recall


def _initialize_dp(
    sequence1_length: int, sequence2_length: int, allocate_pointers: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Initialize dynamic programming data structures for sequence alignment.

    Args:
        sequence1_length: Length of the first sequence.
        sequence2_length: Length of the second sequence.
        allocate_pointers: If True, allocate and initialize the pointers matrix for _traceback.
            If False, return None for pointers. Defaults to True.

    Returns:
        A tuple of (scores matrix, pointers matrix or None).
    """
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))

    if not allocate_pointers:
        return scores, None

    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))

    # Base case: when one sequence is empty, the only option is to skip elements
    # from the other sequence (up = skip seq1, left = skip seq2)
    pointers[1:, 0] = -1
    pointers[0, 1:] = 1

    return scores, pointers


def _traceback(pointers: np.ndarray) -> tuple[list[int], list[int]]:
    """Dynamic programming traceback to determine the aligned indices between two sequences.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left

    Args:
        pointers: DP table of traceback pointers.

    Returns:
        A tuple of (aligned sequence1 indices, aligned sequence2 indices).
    """
    seq1_idx = pointers.shape[0] - 1
    seq2_idx = pointers.shape[1] - 1
    aligned_sequence1_indices: list[int] = []
    aligned_sequence2_indices: list[int] = []

    while not (seq1_idx == 0 and seq2_idx == 0):
        if pointers[seq1_idx, seq2_idx] == -1:
            seq1_idx -= 1
        elif pointers[seq1_idx, seq2_idx] == 1:
            seq2_idx -= 1
        else:
            seq1_idx -= 1
            seq2_idx -= 1
            aligned_sequence1_indices.append(seq1_idx)
            aligned_sequence2_indices.append(seq2_idx)

    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices


def _align_1d(
    sequence1: list[tuple[int, int]],
    sequence2: list[tuple[int, int]],
    reward_lookup: dict[tuple[int, int, int, int], float],
    return_alignment: bool = False,
) -> float | tuple[list[int], list[int], float]:
    """Performs dynamic programming alignment between two sequences using memoized rewards.

    Memoization is an optimization technique that stores previously computed results to avoid redundant calculations.

    Args:
        sequence1: First sequence of index pairs.
        sequence2: Second sequence of index pairs.
        reward_lookup: Lookup table mapping index pairs to reward values.
        return_alignment: If True, returns the alignment indices along with the score. Defaults to False.

    Returns:
        If `return_alignment` is False, returns just the alignment score.
        Otherwise, returns a tuple containing aligned indices of both sequences and the alignment score.
    """
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    # Only allocate pointers when traceback is needed, since this function
    # is called many times from _align_2d_outer with return_alignment=False
    scores, pointers = _initialize_dp(
        sequence1_length, sequence2_length, allocate_pointers=return_alignment
    )

    for seq1_idx in range(1, sequence1_length + 1):
        for seq2_idx in range(1, sequence2_length + 1):
            # Compute reward
            reward = reward_lookup[sequence1[seq1_idx - 1] + sequence2[seq2_idx - 1]]

            # Compute possible alignment scores
            diag_score = scores[seq1_idx - 1, seq2_idx - 1] + reward
            skip_seq2_score = scores[seq1_idx, seq2_idx - 1]
            skip_seq1_score = scores[seq1_idx - 1, seq2_idx]

            # Choose the best alignment option
            max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
            scores[seq1_idx, seq2_idx] = max_score

            if return_alignment:
                assert pointers is not None
                if diag_score == max_score:
                    pointers[seq1_idx, seq2_idx] = 0
                elif skip_seq1_score == max_score:
                    pointers[seq1_idx, seq2_idx] = -1
                else:  # skip_seq2_score == max_score
                    pointers[seq1_idx, seq2_idx] = 1

    score = scores[-1, -1]

    if not return_alignment:
        return score

    assert pointers is not None
    # Traceback to get aligned sequences
    sequence1_indices, sequence2_indices = _traceback(pointers)

    return sequence1_indices, sequence2_indices, score


def _align_2d_outer(
    true_shape: tuple[int, int],
    pred_shape: tuple[int, int],
    reward_lookup: dict[tuple[int, int, int, int], float],
) -> tuple[list[int], list[int], float]:
    """Performs dynamic programming matrix alignment for sequences-of-sequences.

    Aligns two outer sequences, where each entry is also a sequence, using 1D sequence alignment scores.

    Args:
        true_shape: Shape of the true sequence (rows, columns).
        pred_shape: Shape of the predicted sequence (rows, columns).
        reward_lookup: Lookup table mapping index pairs to reward values.

    Returns:
        A tuple containing aligned indices for both sequences and the final alignment score.
    """

    # Pre-build column index sequences for each row to avoid repeated allocation in the inner loop
    true_row_sequences = [
        [(true_row, true_col_idx) for true_col_idx in range(true_shape[1])]
        for true_row in range(true_shape[0])
    ]
    pred_row_sequences = [
        [(pred_row, pred_col_idx) for pred_col_idx in range(pred_shape[1])]
        for pred_row in range(pred_shape[0])
    ]

    # Initialize dynamic programming tables
    scores, pointers = _initialize_dp(true_shape[0], pred_shape[0])
    assert pointers is not None

    for true_row_idx in range(1, true_shape[0] + 1):
        for pred_row_idx in range(1, pred_shape[0] + 1):
            # Compute alignment scores for inner sequences
            reward = _align_1d(
                true_row_sequences[true_row_idx - 1],
                pred_row_sequences[pred_row_idx - 1],
                reward_lookup,
            )

            # Compute possible alignment scores
            diag_score = scores[true_row_idx - 1, pred_row_idx - 1] + reward
            skip_pred_row_score = scores[true_row_idx, pred_row_idx - 1]
            skip_true_row_score = scores[true_row_idx - 1, pred_row_idx]

            # Choose the best alignment option
            max_score = max(diag_score, skip_true_row_score, skip_pred_row_score)
            scores[true_row_idx, pred_row_idx] = max_score

            if diag_score == max_score:
                pointers[true_row_idx, pred_row_idx] = 0
            elif skip_true_row_score == max_score:
                pointers[true_row_idx, pred_row_idx] = -1
            else:
                pointers[true_row_idx, pred_row_idx] = 1

    score = scores[-1, -1]

    # Traceback to get aligned sequences
    aligned_true_indices, aligned_pred_indices = _traceback(pointers)

    return aligned_true_indices, aligned_pred_indices, score


def _get_n_cols(grid: list[list[Any]]) -> int:
    """Determine the number of columns in a 2D grid.

    Args:
        grid: A list of lists representing a 2D grid.

    Returns:
        The number of columns in the grid.
    """
    if len(grid) == 0:
        return 0
    n_cols = len(grid[0])
    if any(len(row) != n_cols for row in grid[1:]):
        raise ValueError("grid must have the same number of columns in each row")
    return n_cols


@dataclasses.dataclass(frozen=True)
class GritsMatchingResult:
    """Result of computing a GriTS metric for a single pair of grids."""

    true_positive_score: float
    true_positive_score_upper_bound: float
    true_grid_scores: np.ndarray
    pred_grid_scores: np.ndarray
    is_exact_match: bool


def factored_2dmss(
    true_grid: np.ndarray, pred_grid: np.ndarray, reward_function: Callable
) -> GritsMatchingResult:
    """Factored 2D-MSS: Factored two-dimensional most-similar substructures

    This is a polynomial-time heuristic to computing the 2D-MSS of two matrices,
    which is NP hard.

    A substructure of a matrix is a subset of its rows and its columns.

    The most similar substructures of two matrices, A and B, are the substructures
    A' and B', where the sum of the similarity over all corresponding entries
    A'(i, j) and B'(i, j) is greatest.

    Args:
        true_grid: Ground truth grid
        pred_grid: Predicted grid
        reward_function: Function to compute similarity between cells

    Returns:
        GritsMatchingResult with the scoring results.
    """
    true_grid_n_rows = len(true_grid)
    true_grid_n_cols = _get_n_cols(true_grid.tolist())
    num_true_cells = true_grid_n_rows * true_grid_n_cols
    pred_grid_n_rows = len(pred_grid)
    pred_grid_n_cols = _get_n_cols(pred_grid.tolist())
    num_pred_cells = pred_grid_n_rows * pred_grid_n_cols

    true_shape = (true_grid_n_rows, true_grid_n_cols)
    pred_shape = (pred_grid_n_rows, pred_grid_n_cols)

    # Early exit: if matrices are the same shape and the elementwise (identity)
    # alignment score exceeds the maximum possible score from any smaller
    # submatrix, the identity alignment is provably optimal—skip the full DP.
    if true_shape == pred_shape:
        grid_scores = np.zeros((true_grid_n_rows, true_grid_n_cols))
        true_positive_score = 0.0
        for true_row, true_col in itertools.product(range(true_shape[0]), range(true_shape[1])):
            cell_score = reward_function(
                true_grid[true_row][true_col], pred_grid[true_row][true_col]
            )
            grid_scores[true_row, true_col] = cell_score
            true_positive_score += cell_score
        if true_positive_score >= (max(true_shape) - 1) * min(true_shape):
            is_exact_match = true_positive_score == num_true_cells == num_pred_cells
            return GritsMatchingResult(
                true_positive_score=true_positive_score,
                true_positive_score_upper_bound=true_positive_score,
                true_grid_scores=grid_scores,
                pred_grid_scores=grid_scores,
                is_exact_match=is_exact_match,
            )

    pre_computed_rewards = {}
    transpose_rewards = {}
    for true_row, true_col, pred_row, pred_col in itertools.product(
        range(true_shape[0]), range(true_shape[1]), range(pred_shape[0]), range(pred_shape[1])
    ):
        reward = reward_function(true_grid[true_row][true_col], pred_grid[pred_row][pred_col])

        pre_computed_rewards[(true_row, true_col, pred_row, pred_col)] = reward
        transpose_rewards[(true_col, true_row, pred_col, pred_row)] = reward

    true_row_nums, pred_row_nums, row_alignment_score = _align_2d_outer(
        true_shape, pred_shape, pre_computed_rewards
    )

    # If matrices have the same number of columns, then after matching/selecting rows,
    # if the resulting match score is already greater than it could be if we removed
    # a row or column from the most similar submatrix, then stop.
    if true_shape[1] == pred_shape[1]:
        true_grid_scores = np.zeros((true_grid_n_rows, true_grid_n_cols))
        pred_grid_scores = np.zeros((pred_grid_n_rows, pred_grid_n_cols))
        true_positive_score = 0.0
        for aligned_row_idx, col in itertools.product(
            range(len(true_row_nums)), range(true_shape[1])
        ):
            true_row = true_row_nums[aligned_row_idx]
            pred_row = pred_row_nums[aligned_row_idx]
            cell_score = reward_function(
                true_grid[true_row][col],
                pred_grid[pred_row][col],
            )
            true_grid_scores[true_row, col] = cell_score
            pred_grid_scores[pred_row, col] = cell_score
            true_positive_score += cell_score
        if true_positive_score >= len(true_row_nums) * true_shape[1] - 1:
            is_exact_match = true_positive_score == num_true_cells == num_pred_cells
            return GritsMatchingResult(
                true_positive_score=true_positive_score,
                true_positive_score_upper_bound=true_positive_score,
                true_grid_scores=true_grid_scores,
                pred_grid_scores=pred_grid_scores,
                is_exact_match=is_exact_match,
            )

    true_column_nums, pred_column_nums, col_alignment_score = _align_2d_outer(
        true_shape[::-1], pred_shape[::-1], transpose_rewards
    )

    true_positive_score_upper_bound = min(row_alignment_score, col_alignment_score)

    true_positive_score = 0.0
    true_grid_scores = np.zeros((true_grid_n_rows, true_grid_n_cols))
    pred_grid_scores = np.zeros((pred_grid_n_rows, pred_grid_n_cols))
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            grid_cell_score = pre_computed_rewards[
                (true_row_num, true_column_num, pred_row_num, pred_column_num)
            ]
            true_grid_scores[true_row_num, true_column_num] = grid_cell_score
            pred_grid_scores[pred_row_num, pred_column_num] = grid_cell_score
            true_positive_score += grid_cell_score

    is_exact_match = true_positive_score == num_true_cells == num_pred_cells
    return GritsMatchingResult(
        true_positive_score=true_positive_score,
        true_positive_score_upper_bound=true_positive_score_upper_bound,
        true_grid_scores=true_grid_scores,
        pred_grid_scores=pred_grid_scores,
        is_exact_match=is_exact_match,
    )


def lcs_similarity(string1: str, string2: str) -> float:
    """Compute similarity between two strings using the longest common subsequence (LCS).

    Args:
        string1: First string to compare.
        string2: Second string to compare.

    Returns:
        float: Similarity score between 0 and 1, computed as
            2 * len(LCS) / (len(string1) + len(string2)).
    """
    if string1 == string2:
        return 1.0

    sequence_matcher = SequenceMatcher(None, string1, string2)

    lcs_length = sum(block.size for block in sequence_matcher.get_matching_blocks())

    lcs_similarity_score = 2 * lcs_length / (len(string1) + len(string2))

    return lcs_similarity_score


def iou(bbox_1_and_area_1: tuple[BBox, float], bbox_2_and_area_2: tuple[BBox, float]) -> float:
    """Compute the intersection-over-union of two bounding boxes.

    Args:
        bbox_1_and_area_1: First bounding box paired with its precomputed area.
        bbox_2_and_area_2: Second bounding box paired with its precomputed area.

    Returns:
        IoU score between 0 and 1.
    """
    bbox1, area_1 = bbox_1_and_area_1
    bbox2, area_2 = bbox_2_and_area_2

    intersection = get_overlap_area(bbox1, bbox2)

    if intersection == 0:
        return 0.0
    union = area_1 + area_2 - intersection
    return intersection / union


def _convert_grid_elements_to_bbox_and_area(grid: list[list]) -> list[list[tuple[BBox, float]]]:
    """Convert grid elements to BBox and area tuples.

    Args:
        grid: Input grid of bounding box coordinates

    Returns:
        Grid of (BBox, area) tuples
    """
    new_grid = []
    for row in grid:
        new_row = []
        for bbox_coords in row:
            bbox = BBox(*bbox_coords)
            new_row.append((bbox, bbox.area))
        new_grid.append(new_row)
    return new_grid


def grits_top_matching(
    true_grid_top: list[list[list[int]]], pred_grid_top: list[list[list[int]]]
) -> GritsMatchingResult:
    """Compute the GriTS_Top true positive score, given two matrices of cell relative spans.

    Args:
        true_grid_top: Ground truth relative span grid
        pred_grid_top: Predicted relative span grid

    Returns:
        GritsMatchingResult with the scoring results.
    """
    return factored_2dmss(
        np.array(_convert_grid_elements_to_bbox_and_area(true_grid_top)),
        np.array(_convert_grid_elements_to_bbox_and_area(pred_grid_top)),
        reward_function=iou,
    )


def grits_loc_matching(
    true_grid_loc: list[list[list[int | float]]], pred_grid_loc: list[list[list[int | float]]]
) -> GritsMatchingResult:
    """Compute the GriTS_Loc true positive score, given two matrices of cell bounding boxes.

    This calculates the spatial location similarity.

    Args:
        true_grid_loc: Ground truth bounding box grid.
        pred_grid_loc: Predicted bounding box grid.

    Returns:
        GritsMatchingResult with the scoring results.
    """

    return factored_2dmss(
        np.array(_convert_grid_elements_to_bbox_and_area(true_grid_loc)),
        np.array(_convert_grid_elements_to_bbox_and_area(pred_grid_loc)),
        reward_function=iou,
    )


def grits_top(
    true_grid_top: list[list[list[int]]], pred_grid_top: list[list[list[int]]]
) -> tuple[float, float, float]:
    """Compute the GriTS_Top F-score, precision, and recall for two relative span grids.

    Args:
        true_grid_top: Ground truth relative span grid
        pred_grid_top: Predicted relative span grid

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = grits_top_matching(true_grid_top, pred_grid_top)
    num_true_cells = len(true_grid_top) * _get_n_cols(true_grid_top)
    num_pred_cells = len(pred_grid_top) * _get_n_cols(pred_grid_top)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


def grits_loc(
    true_grid_loc: list[list[list[int | float]]], pred_grid_loc: list[list[list[int | float]]]
) -> tuple[float, float, float]:
    """Compute the GriTS_Loc F-score, precision, and recall for two bounding box grids.

    Args:
        true_grid_loc: Ground truth bounding box grid.
        pred_grid_loc: Predicted bounding box grid.

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = grits_loc_matching(true_grid_loc, pred_grid_loc)
    num_true_cells = len(true_grid_loc) * _get_n_cols(true_grid_loc)
    num_pred_cells = len(pred_grid_loc) * _get_n_cols(pred_grid_loc)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


def grits_con_matching(
    true_grid_con: list[list[str]], pred_grid_con: list[list[str]]
) -> GritsMatchingResult:
    """Compute the GriTS_Con true positive score, given two matrices of cell text strings.

    Args:
        true_grid_con: Ground truth text grid
        pred_grid_con: Predicted text grid

    Returns:
        GritsMatchingResult with the scoring results.
    """
    return factored_2dmss(
        np.array(true_grid_con, dtype=object),
        np.array(pred_grid_con, dtype=object),
        reward_function=lcs_similarity,
    )


def grits_con(
    true_grid_con: list[list[str]], pred_grid_con: list[list[str]]
) -> tuple[float, float, float]:
    """Compute the GriTS_Con F-score, precision, and recall for two text grids.

    Args:
        true_grid_con: Ground truth text grid
        pred_grid_con: Predicted text grid

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = grits_con_matching(true_grid_con, pred_grid_con)
    num_true_cells = len(true_grid_con) * _get_n_cols(true_grid_con)
    num_pred_cells = len(pred_grid_con) * _get_n_cols(pred_grid_con)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


@dataclasses.dataclass(frozen=True)
class HungarianGritsMatchingResult:
    """Result of computing GriTS metrics across multiple grids using Hungarian matching."""

    true_positive_score: float
    true_positive_score_upper_bound: float
    true_grid_scores: np.ndarray
    pred_grid_scores: np.ndarray
    matched_true_indices: list[int]
    matched_pred_indices: list[int]
    num_exact_grid_matches: int


def _hungarian_grits_matching(
    true_grids: list, pred_grids: list, grits_function: Callable
) -> HungarianGritsMatchingResult:
    """Compute GriTS metrics across multiple grids using Hungarian algorithm for optimal matching.

    Uses the Hungarian algorithm to find the best match between lists of true and predicted grids, and computes the true positive score for the matched pairs.

    Args:
        true_grids: List of ground truth grids
        pred_grids: List of predicted grids
        grits_function: Scoring function to compare a pair of grids (e.g. grits_con_matching, grits_top_matching, or grits_loc_matching)

    Returns:
        HungarianGritsMatchingResult with the matching results.
    """
    num_true_grids = len(true_grids)
    num_pred_grids = len(pred_grids)

    true_grid_scores_matrix = np.empty((num_true_grids, num_pred_grids), dtype=object)
    pred_grid_scores_matrix = np.empty((num_true_grids, num_pred_grids), dtype=object)
    score_matrix = np.empty((num_true_grids, num_pred_grids), dtype=np.float32)
    score_upper_bound_matrix = np.empty((num_true_grids, num_pred_grids), dtype=np.float32)
    is_exact_match_matrix = np.empty((num_true_grids, num_pred_grids), dtype=np.int8)

    for true_idx, true_grid in enumerate(true_grids):
        for pred_idx, pred_grid in enumerate(pred_grids):
            grits_result = grits_function(true_grid, pred_grid)

            true_grid_scores_matrix[true_idx, pred_idx] = grits_result.true_grid_scores
            pred_grid_scores_matrix[true_idx, pred_idx] = grits_result.pred_grid_scores
            score_matrix[true_idx, pred_idx] = grits_result.true_positive_score
            score_upper_bound_matrix[true_idx, pred_idx] = (
                grits_result.true_positive_score_upper_bound
            )
            is_exact_match_matrix[true_idx, pred_idx] = grits_result.is_exact_match

    matched_true_indices, matched_pred_indices = linear_sum_assignment(score_matrix, maximize=True)

    true_grid_scores = true_grid_scores_matrix[matched_true_indices, matched_pred_indices]
    pred_grid_scores = pred_grid_scores_matrix[matched_true_indices, matched_pred_indices]
    true_positive_scores_per_grid = score_matrix[matched_true_indices, matched_pred_indices]
    true_positive_scores_upper_bound = score_upper_bound_matrix[
        matched_true_indices, matched_pred_indices
    ]

    exact_match_by_grid = is_exact_match_matrix[matched_true_indices, matched_pred_indices]
    num_exact_grid_matches = int(sum(exact_match_by_grid))

    true_positive_score = float(true_positive_scores_per_grid.sum())
    true_positive_score_upper_bound = float(true_positive_scores_upper_bound.sum())

    return HungarianGritsMatchingResult(
        true_positive_score=true_positive_score,
        true_positive_score_upper_bound=true_positive_score_upper_bound,
        true_grid_scores=true_grid_scores,
        pred_grid_scores=pred_grid_scores,
        matched_true_indices=list(matched_true_indices),
        matched_pred_indices=list(matched_pred_indices),
        num_exact_grid_matches=num_exact_grid_matches,
    )


def hungarian_grits_top_matching(
    true_grids_top: list[list[list[list[int]]]], pred_grids_top: list[list[list[list[int]]]]
) -> HungarianGritsMatchingResult:
    """Compute GriTS_Top metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_top: List of ground truth relative span grids
        pred_grids_top: List of predicted relative span grids

    Returns:
        HungarianGritsMatchingResult with the matching results.
    """
    return _hungarian_grits_matching(true_grids_top, pred_grids_top, grits_top_matching)


def hungarian_grits_top(
    true_grids_top: list[list[list[list[int]]]], pred_grids_top: list[list[list[list[int]]]]
) -> tuple[float, float, float]:
    """Compute the GriTS_Top F-score, precision, and recall across multiple relative span grids.

    Uses the Hungarian algorithm to find the optimal one-to-one matching between
    true and predicted grids before computing the aggregate score.

    Args:
        true_grids_top: List of ground truth relative span grids
        pred_grids_top: List of predicted relative span grids

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = hungarian_grits_top_matching(true_grids_top, pred_grids_top)
    num_true_cells = sum(len(g) * _get_n_cols(g) for g in true_grids_top)
    num_pred_cells = sum(len(g) * _get_n_cols(g) for g in pred_grids_top)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


def hungarian_grits_con_matching(
    true_grids_con: list[list[list[str]]], pred_grids_con: list[list[list[str]]]
) -> HungarianGritsMatchingResult:
    """Compute GriTS_Con metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_con: List of ground truth text content grids
        pred_grids_con: List of predicted text content grids

    Returns:
        HungarianGritsMatchingResult with the matching results.
    """
    return _hungarian_grits_matching(true_grids_con, pred_grids_con, grits_con_matching)


def hungarian_grits_con(
    true_grids_con: list[list[list[str]]], pred_grids_con: list[list[list[str]]]
) -> tuple[float, float, float]:
    """Compute the GriTS_Con F-score, precision, and recall across multiple text grids.

    Uses the Hungarian algorithm to find the optimal one-to-one matching between
    true and predicted grids before computing the aggregate score.

    Args:
        true_grids_con: List of ground truth text content grids
        pred_grids_con: List of predicted text content grids

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = hungarian_grits_con_matching(true_grids_con, pred_grids_con)
    num_true_cells = sum(len(g) * _get_n_cols(g) for g in true_grids_con)
    num_pred_cells = sum(len(g) * _get_n_cols(g) for g in pred_grids_con)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


def hungarian_grits_loc_matching(
    true_grids_loc: list[list[list[list[int | float]]]],
    pred_grids_loc: list[list[list[list[int | float]]]],
) -> HungarianGritsMatchingResult:
    """Compute GriTS_Loc metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_loc: List of ground truth bounding box grids
        pred_grids_loc: List of predicted bounding box grids

    Returns:
        HungarianGritsMatchingResult with the matching results.
    """
    return _hungarian_grits_matching(true_grids_loc, pred_grids_loc, grits_loc_matching)


def hungarian_grits_loc(
    true_grids_loc: list[list[list[list[int | float]]]],
    pred_grids_loc: list[list[list[list[int | float]]]],
) -> tuple[float, float, float]:
    """Compute the GriTS_Loc F-score, precision, and recall across multiple bounding box grids.

    Uses the Hungarian algorithm to find the optimal one-to-one matching between
    true and predicted grids before computing the aggregate score.

    Args:
        true_grids_loc: List of ground truth bounding box grids
        pred_grids_loc: List of predicted bounding box grids

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = hungarian_grits_loc_matching(true_grids_loc, pred_grids_loc)
    num_true_cells = sum(len(g) * _get_n_cols(g) for g in true_grids_loc)
    num_pred_cells = sum(len(g) * _get_n_cols(g) for g in pred_grids_loc)
    return compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)


class _GritsMetricEvaluator:
    """Accumulates results from GriTS evaluation calls for a single metric and computes aggregate scores."""

    _METRIC_FUNCTIONS: dict[str, Callable[[list, list], HungarianGritsMatchingResult]] = {
        "con": hungarian_grits_con_matching,
        "loc": hungarian_grits_loc_matching,
        "top": hungarian_grits_top_matching,
    }

    def __init__(self, metric: str = "con") -> None:
        """Initialize the evaluator with the specified metric."""
        if metric not in self._METRIC_FUNCTIONS:
            raise ValueError(f"metric must be one of {set(self._METRIC_FUNCTIONS)}, got {metric!r}")
        self._metric = metric
        self._grits_function: Callable[[list, list], HungarianGritsMatchingResult] = (
            self._METRIC_FUNCTIONS[metric]
        )
        self.true_positive_scores: list[float] = []
        self.num_true_cells: list[int] = []
        self.num_pred_cells: list[int] = []
        self.num_true_grids: list[int] = []
        self.num_pred_grids: list[int] = []
        self.num_exact_grid_matches: list[int] = []
        self.num_exact_cell_matches: list[int] = []

    def eval_grids(self, true_grids: list, pred_grids: list) -> None:
        """Evaluate a sample and store the results."""
        result = self._grits_function(true_grids, pred_grids)
        true_positive_score = result.true_positive_score
        true_grid_scores = result.true_grid_scores
        num_exact_grid_matches = result.num_exact_grid_matches
        num_true_cells = sum([len(true_grid) * _get_n_cols(true_grid) for true_grid in true_grids])
        num_pred_cells = sum([len(pred_grid) * _get_n_cols(pred_grid) for pred_grid in pred_grids])
        num_exact_cell_matches = sum(int((scores == 1).sum()) for scores in true_grid_scores)
        self.true_positive_scores.append(true_positive_score)
        self.num_true_cells.append(num_true_cells)
        self.num_pred_cells.append(num_pred_cells)
        self.num_true_grids.append(len(true_grids))
        self.num_pred_grids.append(len(pred_grids))
        self.num_exact_grid_matches.append(num_exact_grid_matches)
        self.num_exact_cell_matches.append(num_exact_cell_matches)

    def compute_grits(self) -> dict[str, float]:
        """Compute the (micro) GriTS score, which is a pseudo F1 score over all instances.

        Returns:
            Dict with keys grits_METRIC, grits_METRIC_precision, grits_METRIC_recall,
            grits_METRIC_cell_exact_match_accuracy, and grits_METRIC_grid_exact_match_accuracy.
        """
        total_true_positive_score = sum(self.true_positive_scores)
        total_num_true = sum(self.num_true_cells)
        total_num_positives = sum(self.num_pred_cells)
        f1, precision, recall = compute_fscore(
            total_true_positive_score, total_num_true, total_num_positives
        )
        cell_exact_match_accuracy = (
            sum(self.num_exact_cell_matches) / total_num_true
            if self.num_exact_cell_matches
            else 0.0
        )
        grid_exact_match_accuracy = (
            sum(self.num_exact_grid_matches) / sum(self.num_true_grids)
            if self.num_exact_grid_matches
            else 0.0
        )
        m = self._metric
        return {
            f"grits_{m}": f1,
            f"grits_{m}_precision": precision,
            f"grits_{m}_recall": recall,
            f"grits_{m}_cell_exact_match_accuracy": cell_exact_match_accuracy,
            f"grits_{m}_grid_exact_match_accuracy": grid_exact_match_accuracy,
        }

    def compute_avg_grits(self) -> dict[str, float]:
        """Compute the (macro) average GriTS score. Compute the GriTS score for each instance, then average the GriTS score over all instances.

        Returns:
            Dict with keys avg_grits_METRIC, avg_grits_METRIC_precision,
            avg_grits_METRIC_recall, avg_grits_METRIC_cell_exact_match_accuracy,
            and avg_grits_METRIC_grid_exact_match_accuracy.
        """
        f1s = []
        precisions = []
        recalls = []
        cell_exact_match_accuracies = []
        grid_exact_match_accuracies = []
        for (
            true_positive_score,
            num_true_cells,
            num_positive,
            num_true_grids,
            exact_cell_matches,
            exact_grid_matches,
        ) in zip(
            self.true_positive_scores,
            self.num_true_cells,
            self.num_pred_cells,
            self.num_true_grids,
            self.num_exact_cell_matches,
            self.num_exact_grid_matches,
        ):
            f1, precision, recall = compute_fscore(
                true_positive_score, num_true_cells, num_positive
            )
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            cell_exact_match_accuracies.append(exact_cell_matches / num_true_cells)
            grid_exact_match_accuracies.append(exact_grid_matches / num_true_grids)
        m = self._metric
        return {
            f"avg_grits_{m}": sum(f1s) / len(f1s),
            f"avg_grits_{m}_precision": sum(precisions) / len(precisions),
            f"avg_grits_{m}_recall": sum(recalls) / len(recalls),
            f"avg_grits_{m}_cell_exact_match_accuracy": sum(cell_exact_match_accuracies)
            / len(cell_exact_match_accuracies),
            f"avg_grits_{m}_grid_exact_match_accuracy": sum(grid_exact_match_accuracies)
            / len(grid_exact_match_accuracies),
        }


class GritsEvaluator:
    """Tracks multiple GriTS metrics simultaneously by composing single-metric evaluators."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialize the evaluator with the specified metrics.

        Args:
            metrics: List of metric names to track. Defaults to ["con", "loc", "top"].
        """
        if metrics is None:
            metrics = ["con", "loc", "top"]
        self._evaluators = {metric: _GritsMetricEvaluator(metric=metric) for metric in metrics}

    @property
    def metrics(self) -> list[str]:
        """Return the list of metric names being tracked."""
        return list(self._evaluators)

    def eval_grids_by_metric(
        self,
        true_grids: dict[str, list],
        pred_grids: dict[str, list],
    ) -> None:
        """Evaluate a sample for each metric using a dict of grids.

        Args:
            true_grids: Dict mapping metric name to list of ground truth grids.
            pred_grids: Dict mapping metric name to list of predicted grids.
        """
        for metric, evaluator in self._evaluators.items():
            evaluator.eval_grids(true_grids[metric], pred_grids[metric])

    def eval_grids(
        self,
        true_grids: list,
        pred_grids: list,
    ) -> None:
        """Evaluate a sample for a single metric using lists of grids.

        Args:
            true_grids: List of ground truth grids.
            pred_grids: List of predicted grids.

        Raises:
            TypeError: If the evaluator tracks more than one metric.
        """
        if len(self._evaluators) != 1:
            raise TypeError(
                "eval_grids requires the evaluator to track exactly one metric, "
                f"but it tracks {list(self._evaluators)}. "
                "Use eval_grids_by_metric with a dict mapping metric name to list of grids."
            )
        metric = next(iter(self._evaluators))
        self._evaluators[metric].eval_grids(true_grids, pred_grids)

    def eval_htmls(self, true_htmls: list[str], pred_htmls: list[str]) -> None:
        """Parse HTML tables and evaluate the con and top metrics.

        Args:
            true_htmls: List of HTML strings of ground truth tables.
            pred_htmls: List of HTML strings of predicted tables.

        Raises:
            TypeError: If a bare string is passed instead of a list of strings.
            ValueError: If any HTML string cannot be parsed, or if the evaluator
                is not tracking both "con" and "top" metrics.
        """
        if isinstance(true_htmls, str) or isinstance(pred_htmls, str):  # type: ignore[unreachable]
            raise TypeError("eval_htmls expects lists of HTML strings, not a single string.")
        missing = {"con", "top"} - set(self._evaluators)
        if missing:
            raise ValueError(
                f"eval_htmls requires both 'con' and 'top' metrics, but missing: {missing}"
            )

        true_grids_con = []
        true_grids_top = []
        for true_html in true_htmls:
            true_cell_dicts = html_to_cell_dicts(true_html)
            if true_cell_dicts is None:
                raise ValueError(f"Failed to parse true_html: {true_html!r}")
            true_grids_con.append(cell_dicts_to_grid_con(true_cell_dicts))
            true_grids_top.append(cell_dicts_to_grid_top(true_cell_dicts))

        pred_grids_con = []
        pred_grids_top = []
        for pred_html in pred_htmls:
            pred_cell_dicts = html_to_cell_dicts(pred_html)
            if pred_cell_dicts is None:
                raise ValueError(f"Failed to parse pred_html: {pred_html!r}")
            pred_grids_con.append(cell_dicts_to_grid_con(pred_cell_dicts))
            pred_grids_top.append(cell_dicts_to_grid_top(pred_cell_dicts))

        self._evaluators["con"].eval_grids(true_grids_con, pred_grids_con)
        self._evaluators["top"].eval_grids(true_grids_top, pred_grids_top)

    def eval_table_cell_dicts(
        self,
        true_cell_dicts: list[list[dict[str, Any]]],
        pred_cell_dicts: list[list[dict[str, Any]]],
    ) -> None:
        """Convert table cell dicts to grids and evaluate all three metrics (con, loc, top).

        Args:
            true_cell_dicts: A list of lists of cell dicts, one list per ground truth table.
            pred_cell_dicts: A list of lists of cell dicts, one list per predicted table.

        Raises:
            TypeError: If a flat list of cell dicts is passed instead of a list of lists.
            ValueError: If the evaluator is not tracking all three metrics.
        """
        if true_cell_dicts and isinstance(true_cell_dicts[0], dict):  # type: ignore[unreachable]
            raise TypeError(
                "eval_table_cell_dicts expects a list of lists of cell dicts, "
                "not a single list of cell dicts."
            )
        if pred_cell_dicts and isinstance(pred_cell_dicts[0], dict):  # type: ignore[unreachable]
            raise TypeError(
                "eval_table_cell_dicts expects a list of lists of cell dicts, "
                "not a single list of cell dicts."
            )
        missing = {"con", "loc", "top"} - set(self._evaluators)
        if missing:
            raise ValueError(
                f"eval_table_cell_dicts requires 'con', 'loc', and 'top' metrics, but missing: {missing}"
            )

        true_grids_con = []
        true_grids_loc = []
        true_grids_top = []
        for cell_dicts in true_cell_dicts:
            true_grids_con.append(cell_dicts_to_grid_con(cell_dicts))
            true_grids_loc.append(cell_dicts_to_grid_loc(cell_dicts))
            true_grids_top.append(cell_dicts_to_grid_top(cell_dicts))

        pred_grids_con = []
        pred_grids_loc = []
        pred_grids_top = []
        for cell_dicts in pred_cell_dicts:
            pred_grids_con.append(cell_dicts_to_grid_con(cell_dicts))
            pred_grids_loc.append(cell_dicts_to_grid_loc(cell_dicts))
            pred_grids_top.append(cell_dicts_to_grid_top(cell_dicts))

        self._evaluators["con"].eval_grids(true_grids_con, pred_grids_con)
        self._evaluators["loc"].eval_grids(true_grids_loc, pred_grids_loc)
        self._evaluators["top"].eval_grids(true_grids_top, pred_grids_top)

    def _check_sample_counts(self) -> None:
        """Raise ValueError if evaluated metrics have different numbers of samples."""
        counts = {
            m: len(ev.true_positive_scores)
            for m, ev in self._evaluators.items()
            if len(ev.true_positive_scores) > 0
        }
        if len(set(counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across metrics: {counts}")

    def compute_grits(self) -> dict[str, float]:
        """Compute micro GriTS scores for each metric.

        Returns:
            Dict with keys like grits_con, grits_con_precision, grits_con_recall
            for each metric with at least one evaluated sample.

        Raises:
            ValueError: If evaluated metrics have different numbers of samples.
        """
        self._check_sample_counts()
        results: dict[str, float] = {}
        for ev in self._evaluators.values():
            if len(ev.true_positive_scores) > 0:
                results.update(ev.compute_grits())
        return results

    def compute_avg_grits(self) -> dict[str, float]:
        """Compute macro-averaged GriTS scores for each metric.

        Returns:
            Dict with keys like avg_grits_con, avg_grits_con_precision, avg_grits_con_recall
            for each metric with at least one evaluated sample.

        Raises:
            ValueError: If evaluated metrics have different numbers of samples.
        """
        self._check_sample_counts()
        results: dict[str, float] = {}
        for ev in self._evaluators.values():
            if len(ev.true_positive_scores) > 0:
                results.update(ev.compute_avg_grits())
        return results
