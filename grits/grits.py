# Copyright 2025-present Kensho Technologies, LLC.
from difflib import SequenceMatcher
import itertools
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bbox import BBox, get_overlap_area


def cells_to_grid(
    cells: list[dict[str, Any]], key: str = "bbox", fill_missing: bool = False
) -> list[list[Any]]:
    """Convert from a list of cells to a matrix of grid cell features.

    This matrix representation is the input to GriTS.

    For key, use:
    - 'bbox' for computing GriTS_Loc
    - 'cell_text' for computing GriTS_Con

    Args:
        cells: List of cell dicts with 'row_nums', 'column_nums', and feature keys.
        key: The cell feature key to extract ('bbox' for GriTS_Loc, 'cell_text' for GriTS_Con).
        fill_missing: If True, initialize missing cells with empty defaults instead of zeros.

    Returns:
        A 2D list (grid) of cell features indexed by [row][column].
    """
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    if fill_missing:
        # Initialize with empty strings for text or empty list for bbox
        # This handles cases where HTML tables have missing cells
        default_value = "" if key == "cell_text" else []
        cell_grid = [[default_value for _ in range(num_columns)] for _ in range(num_rows)]
    else:
        cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
                cell_grid[row_num][column_num] = cell[key]

    return cell_grid


def cells_to_grid_top(
    cells: list[dict[str, Any]], fill_missing: bool = False
) -> list[list[Any]]:
    """Convert from a list of cells to the matrix/grid of relative spans (boxes), used for computing GriTS_Top. Previously known as cells_to_relspan_grid().

    Args:
        cells: List of cell dicts with 'row_nums' and 'column_nums'.
        fill_missing: If True, initialize missing cells with [0, 0, 1, 1] instead of zeros.

    Returns:
        A 2D list (grid) of relative span values indexed by [row][column].
    """
    if len(cells) == 0:
        return [[]]
    num_rows = max([max(cell["row_nums"]) for cell in cells]) + 1
    num_columns = max([max(cell["column_nums"]) for cell in cells]) + 1
    if fill_missing:
        # Initialize grid with [0, 0, 1, 1] (non-spanning cells) instead of 0.0
        # This handles cases where HTML tables have missing cells
        cell_grid = [[[0, 0, 1, 1] for _ in range(num_columns)] for _ in range(num_rows)]
    else:
        cell_grid = np.zeros((num_rows, num_columns)).tolist()
    for cell in cells:
        min_row_num = min(cell["row_nums"])
        min_column_num = min(cell["column_nums"])
        max_row_num = max(cell["row_nums"]) + 1
        max_column_num = max(cell["column_nums"]) + 1
        for row_num in cell["row_nums"]:
            for column_num in cell["column_nums"]:
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


def initialize_DP(
    sequence1_length: int, sequence2_length: int, allocate_pointers: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Initialize dynamic programming data structures for sequence alignment.

    Args:
        sequence1_length: Length of the first sequence.
        sequence2_length: Length of the second sequence.
        allocate_pointers: If True, allocate and initialize the pointers matrix for traceback.
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


def traceback(pointers: np.ndarray) -> tuple[list[int], list[int]]:
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


def align_1d(
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
    # is called many times from align_2d_outer with return_alignment=False
    scores, pointers = initialize_DP(
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
    sequence1_indices, sequence2_indices = traceback(pointers)

    return sequence1_indices, sequence2_indices, score


def align_2d_outer(
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
    scores, pointers = initialize_DP(true_shape[0], pred_shape[0])
    assert pointers is not None

    for true_row_idx in range(1, true_shape[0] + 1):
        for pred_row_idx in range(1, pred_shape[0] + 1):
            # Compute alignment scores for inner sequences
            reward = align_1d(
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
    aligned_true_indices, aligned_pred_indices = traceback(pointers)

    return aligned_true_indices, aligned_pred_indices, score


def _get_n_cols(grid: list[list[object]]) -> int:
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


def factored_2dmss(
    true_cell_grid: np.ndarray, pred_cell_grid: np.ndarray, reward_function: Callable
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Factored 2D-MSS: Factored two-dimensional most-similar substructures

    This is a polynomial-time heuristic to computing the 2D-MSS of two matrices,
    which is NP hard.

    A substructure of a matrix is a subset of its rows and its columns.

    The most similar substructures of two matrices, A and B, are the substructures
    A' and B', where the sum of the similarity over all corresponding entries
    A'(i, j) and B'(i, j) is greatest.

    Args:
        true_cell_grid: Ground truth grid
        pred_cell_grid: Predicted grid
        reward_function: Function to compute similarity between cells

    Returns:
        Tuple of (true_positive_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores), where the scalar values are floats
        and the grid scores are numpy arrays.
    """
    true_cell_grid_n_rows = len(true_cell_grid)
    true_cell_grid_n_cols = _get_n_cols(true_cell_grid.tolist())
    pred_cell_grid_n_rows = len(pred_cell_grid)
    pred_cell_grid_n_cols = _get_n_cols(pred_cell_grid.tolist())

    true_shape = (true_cell_grid_n_rows, true_cell_grid_n_cols)
    pred_shape = (pred_cell_grid_n_rows, pred_cell_grid_n_cols)

    # Early exit: if matrices are the same shape and the elementwise (identity)
    # alignment score exceeds the maximum possible score from any smaller
    # submatrix, the identity alignment is provably optimal—skip the full DP.
    if true_shape == pred_shape:
        grid_scores = np.zeros((true_cell_grid_n_rows, true_cell_grid_n_cols))
        true_positive_score = 0.0
        for true_row, true_col in itertools.product(range(true_shape[0]), range(true_shape[1])):
            cell_score = reward_function(
                true_cell_grid[true_row][true_col], pred_cell_grid[true_row][true_col]
            )
            grid_scores[true_row, true_col] = cell_score
            true_positive_score += cell_score
        if true_positive_score >= (max(true_shape) - 1) * min(true_shape):
            return true_positive_score, true_positive_score, grid_scores, grid_scores

    pre_computed_rewards = {}
    transpose_rewards = {}
    for true_row, true_col, pred_row, pred_col in itertools.product(
        range(true_shape[0]), range(true_shape[1]), range(pred_shape[0]), range(pred_shape[1])
    ):
        reward = reward_function(
            true_cell_grid[true_row][true_col], pred_cell_grid[pred_row][pred_col]
        )

        pre_computed_rewards[(true_row, true_col, pred_row, pred_col)] = reward
        transpose_rewards[(true_col, true_row, pred_col, pred_row)] = reward

    true_row_nums, pred_row_nums, row_alignment_score = align_2d_outer(
        true_shape, pred_shape, pre_computed_rewards
    )

    # If matrices have the same number of columns, then after matching/selecting rows,
    # if the resulting match score is already greater than it could be if we removed
    # a row or column from the most similar submatrix, then stop.
    if true_shape[1] == pred_shape[1]:
        true_grid_scores = np.zeros((true_cell_grid_n_rows, true_cell_grid_n_cols))
        pred_grid_scores = np.zeros((pred_cell_grid_n_rows, pred_cell_grid_n_cols))
        true_positive_score = 0.0
        for aligned_row_idx, col in itertools.product(
            range(len(true_row_nums)), range(true_shape[1])
        ):
            true_row = true_row_nums[aligned_row_idx]
            pred_row = pred_row_nums[aligned_row_idx]
            cell_score = reward_function(
                true_cell_grid[true_row][col],
                pred_cell_grid[pred_row][col],
            )
            true_grid_scores[true_row, col] = cell_score
            pred_grid_scores[pred_row, col] = cell_score
            true_positive_score += cell_score
        if true_positive_score >= len(true_row_nums) * true_shape[1] - 1:
            return true_positive_score, true_positive_score, true_grid_scores, pred_grid_scores

    true_column_nums, pred_column_nums, col_alignment_score = align_2d_outer(
        true_shape[::-1], pred_shape[::-1], transpose_rewards
    )

    true_positive_score_upper_bound = min(row_alignment_score, col_alignment_score)

    true_positive_score = 0.0
    true_grid_scores = np.zeros((true_cell_grid_n_rows, true_cell_grid_n_cols))
    pred_grid_scores = np.zeros((pred_cell_grid_n_rows, pred_cell_grid_n_cols))
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            grid_cell_score = pre_computed_rewards[
                (true_row_num, true_column_num, pred_row_num, pred_column_num)
            ]
            true_grid_scores[true_row_num, true_column_num] = grid_cell_score
            pred_grid_scores[pred_row_num, pred_column_num] = grid_cell_score
            true_positive_score += grid_cell_score

    return true_positive_score, true_positive_score_upper_bound, true_grid_scores, pred_grid_scores


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


def grits_top(
    true_relative_span_grid: list[list[list[int]]], pred_relative_span_grid: list[list[list[int]]]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Top given two matrices of cell relative spans.

    For the cell at grid location (i,j), let a(i,j) be its rowspan,
    let β(i,j) be its colspan, let p(i,j) be the minimum row it occupies,
    and let θ(i,j) be the minimum column it occupies. Its relative span is
    bounding box [θ(i,j)-j, p(i,j)-i, θ(i,j)-j+β(i,j), p(i,j)-i+a(i,j)].

    It gives the size and location of the cell each grid cell belongs to
    relative to the current grid cell location, in grid coordinate units.
    Note that for a non-spanning cell this will always be [0, 0, 1, 1].

    Args:
        true_relative_span_grid: Ground truth relative span grid
        pred_relative_span_grid: Predicted relative span grid

    Returns:
        Tuple of (true_positive_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores).
    """
    return factored_2dmss(
        np.array(_convert_grid_elements_to_bbox_and_area(true_relative_span_grid)),
        np.array(_convert_grid_elements_to_bbox_and_area(pred_relative_span_grid)),
        reward_function=iou,
    )


def grits_loc(
    true_bbox_grid: list[list[list[int | float]]], pred_bbox_grid: list[list[list[int | float]]]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Loc given two matrices of cell bounding boxes.

    This calculates the spatial location similarity.

    Args:
        true_bbox_grid: Ground truth bounding box grid.
        pred_bbox_grid: Predicted bounding box grid.

    Returns:
        Tuple of (true_positive_score, true_positive_score_upper_bound, true_grid_scores, pred_grid_scores).
    """

    return factored_2dmss(
        np.array(_convert_grid_elements_to_bbox_and_area(true_bbox_grid)),
        np.array(_convert_grid_elements_to_bbox_and_area(pred_bbox_grid)),
        reward_function=iou,
    )


def grits_con(
    true_text_grid: list[list[str]], pred_text_grid: list[list[str]]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Con given two matrices of cell text strings.

    Args:
        true_text_grid: Ground truth text grid
        pred_text_grid: Predicted text grid

    Returns:
        Tuple of (true_positive_score, true_positive_score_upper_bound, true_grid_scores, pred_grid_scores).
    """
    return factored_2dmss(
        np.array(true_text_grid, dtype=object),
        np.array(pred_text_grid, dtype=object),
        reward_function=lcs_similarity,
    )


def _hungarian_grits(
    true_grids: list, pred_grids: list, grits_function: Callable
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS metrics across multiple grids using Hungarian algorithm for optimal matching.

    Uses the Hungarian algorithm to find the best matching between true and predicted grids,
    then computes aggregated GriTS metrics across all matched pairs.

    Args:
        true_grids: List of ground truth grids
        pred_grids: List of predicted grids
        grits_function: Scoring function to compare a pair of grids (e.g. grits_top or grits_con)

    Returns:
        Tuple of (true_positive_cell_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores).
    """
    num_true_grids = len(true_grids)
    num_pred_grids = len(pred_grids)

    true_grid_scores_matrix = np.empty((num_true_grids, num_pred_grids), dtype=object)
    pred_grid_scores_matrix = np.empty((num_true_grids, num_pred_grids), dtype=object)
    score_matrix = np.empty((num_true_grids, num_pred_grids), dtype=np.float32)
    score_upper_bound_matrix = np.empty((num_true_grids, num_pred_grids), dtype=np.float32)

    for true_idx, true_grid in enumerate(true_grids):
        for pred_idx, pred_grid in enumerate(pred_grids):
            (
                true_positive_cell_score,
                true_positive_cell_score_upper_bound,
                true_grid_scores,
                pred_grid_scores,
            ) = grits_function(true_grid, pred_grid)

            true_grid_scores_matrix[true_idx, pred_idx] = true_grid_scores
            pred_grid_scores_matrix[true_idx, pred_idx] = pred_grid_scores
            score_matrix[true_idx, pred_idx] = true_positive_cell_score
            score_upper_bound_matrix[true_idx, pred_idx] = true_positive_cell_score_upper_bound

    matched_true_indices, matched_pred_indices = linear_sum_assignment(score_matrix, maximize=True)

    true_grid_scores = true_grid_scores_matrix[matched_true_indices, matched_pred_indices]
    pred_grid_scores = pred_grid_scores_matrix[matched_true_indices, matched_pred_indices]
    true_positive_scores_per_grid = score_matrix[matched_true_indices, matched_pred_indices]
    true_positive_scores_upper_bound = score_upper_bound_matrix[
        matched_true_indices, matched_pred_indices
    ]

    true_positive_score = float(true_positive_scores_per_grid.sum())
    true_positive_score_upper_bound = float(true_positive_scores_upper_bound.sum())

    return true_positive_score, true_positive_score_upper_bound, true_grid_scores, pred_grid_scores


def hungarian_grits_top(
    true_grids_top: list[list[list[list[int]]]], pred_grids_top: list[list[list[list[int]]]]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Top metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_top: List of ground truth relative span grids
        pred_grids_top: List of predicted relative span grids

    Returns:
        Tuple of (true_positive_cell_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores).
    """
    return _hungarian_grits(true_grids_top, pred_grids_top, grits_top)


def hungarian_grits_con(
    true_grids_con: list[list[list[str]]], pred_grids_con: list[list[list[str]]]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Con metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_con: List of ground truth text content grids
        pred_grids_con: List of predicted text content grids

    Returns:
        Tuple of (true_positive_cell_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores).
    """
    return _hungarian_grits(true_grids_con, pred_grids_con, grits_con)


def hungarian_grits_loc(
    true_grids_loc: list[list[list[list[int | float]]]],
    pred_grids_loc: list[list[list[list[int | float]]]],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute GriTS_Loc metrics across multiple grids using Hungarian algorithm for optimal matching.

    Args:
        true_grids_loc: List of ground truth bounding box grids
        pred_grids_loc: List of predicted bounding box grids

    Returns:
        Tuple of (true_positive_cell_score, true_positive_score_upper_bound,
        true_grid_scores, pred_grid_scores).
    """
    return _hungarian_grits(true_grids_loc, pred_grids_loc, grits_loc)


class GritsEvaluator:
    """Accumulates results from multiple grits calls and computes aggregate F1."""

    _METRIC_FUNCTIONS: dict[
        str, Callable[[list, list], tuple[float, float, np.ndarray, np.ndarray]]
    ] = {
        "con": hungarian_grits_con,
        "loc": hungarian_grits_loc,
        "top": hungarian_grits_top,
    }

    def __init__(self, metric: str = "con") -> None:
        """Initialize the evaluator with the specified metric."""
        if metric not in self._METRIC_FUNCTIONS:
            raise ValueError(f"metric must be one of {set(self._METRIC_FUNCTIONS)}, got {metric!r}")
        self._grits_function: Callable[
            [list, list], tuple[float, float, np.ndarray, np.ndarray]
        ] = self._METRIC_FUNCTIONS[metric]
        self.true_positive_scores: list[float] = []
        self.num_true_cells: list[int] = []
        self.num_pred_cells: list[int] = []

    def eval_sample(self, true_grids: list | np.ndarray, pred_grids: list | np.ndarray) -> None:
        """Evaluate a sample and store the results."""
        if isinstance(true_grids, np.ndarray):
            true_grids = [true_grids]
        if isinstance(pred_grids, np.ndarray):
            pred_grids = [pred_grids]
        result = self._grits_function(true_grids, pred_grids)
        true_positive_score = result[0]
        num_true = sum([len(true_grid) * _get_n_cols(true_grid) for true_grid in true_grids])
        num_pred = sum([len(pred_grid) * _get_n_cols(pred_grid) for pred_grid in pred_grids])
        self.true_positive_scores.append(true_positive_score)
        self.num_true_cells.append(num_true)
        self.num_pred_cells.append(num_pred)

    def compute_grits(self) -> tuple[float, float, float]:
        """Compute the (micro) GriTS score, which is a pseudo F1 score over all instances.

        Returns:
            A tuple containing (GriTS F1-score, Precision, Recall).
        """
        total_true_positive_score = sum(self.true_positive_scores)
        total_num_true = sum(self.num_true_cells)
        total_num_positives = sum(self.num_pred_cells)
        return compute_fscore(total_true_positive_score, total_num_true, total_num_positives)

    def compute_avg_grits(self) -> tuple[float, float, float]:
        """Compute the (macro) average GriTS score. Compute the GriTS score for each instance, then average the GriTS score over all instances.

        Returns:
            A tuple containing (GriTS F1-score, Precision, Recall).
        """

        f1s = []
        precisions = []
        recalls = []
        for true_positive_score, num_true, num_positive in zip(
            self.true_positive_scores, self.num_true_cells, self.num_pred_cells
        ):
            f1, precision, recall = compute_fscore(true_positive_score, num_true, num_positive)
            f1s.append(f1)
            recalls.append(recall)
            precisions.append(precision)

        avg_f1 = sum(f1s) / len(f1s)
        avg_recall = sum(recalls) / len(recalls)
        avg_precision = sum(precisions) / len(precisions)

        return avg_f1, avg_precision, avg_recall
