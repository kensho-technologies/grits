# Copyright 2025-present Kensho Technologies, LLC.
import numpy as np
import pytest

from grits.bbox import BBox
from grits.grits import (
    GritsEvaluator,
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
from grits.utils import html_to_cells


def _make_simple_cells():
    """Return a simple 2x2 table as a list of cell dicts."""
    return [
        {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
        {"row_nums": [0], "column_nums": [1], "cell_text": "B", "bbox": [10, 0, 20, 10]},
        {"row_nums": [1], "column_nums": [0], "cell_text": "C", "bbox": [0, 10, 10, 20]},
        {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
    ]


class TestCellsToGridCon:
    def test_empty_cells(self):
        assert cells_to_grid_con([]) == [[]]

    def test_simple_text_grid(self):
        cells = _make_simple_cells()
        grid = cells_to_grid_con(cells)
        assert grid == [["A", "B"], ["C", "D"]]

    def test_spanning_cell(self):
        cells = [
            {"row_nums": [0, 1], "column_nums": [0], "cell_text": "span", "bbox": [0, 0, 10, 20]},
            {"row_nums": [0], "column_nums": [1], "cell_text": "B", "bbox": [10, 0, 20, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cells_to_grid_con(cells)
        assert grid == [["span", "B"], ["span", "D"]]

    def test_missing_cells(self):
        cells = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cells_to_grid_con(cells)
        assert grid == [["A", ""], ["", "D"]]


class TestCellsToGridLoc:
    def test_empty_cells(self):
        assert cells_to_grid_loc([]) == [[]]

    def test_simple_bbox_grid(self):
        cells = _make_simple_cells()
        grid = cells_to_grid_loc(cells)
        assert grid == [
            [[0, 0, 10, 10], [10, 0, 20, 10]],
            [[0, 10, 10, 20], [10, 10, 20, 20]],
        ]

    def test_missing_cells(self):
        cells = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cells_to_grid_loc(cells)
        assert grid[0][0] == [0, 0, 10, 10]
        assert grid[0][1] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][0] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][1] == [10, 10, 20, 20]


class TestCellsToGridTop:
    def test_empty_cells(self):
        assert cells_to_grid_top([]) == [[]]

    def test_non_spanning(self):
        cells = _make_simple_cells()
        grid = cells_to_grid_top(cells)
        assert grid == [
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
        ]

    def test_spanning_cell(self):
        cells = [
            {"row_nums": [0, 1], "column_nums": [0, 1], "cell_text": "big"},
        ]
        grid = cells_to_grid_top(cells)
        # top-left: min_col=0, min_row=0, max_col=2, max_row=2
        assert grid[0][0] == [0, 0, 2, 2]
        # top-right (col=1): min_col - 1 = -1, min_row - 0 = 0, max_col - 1 = 1, max_row - 0 = 2
        assert grid[0][1] == [-1, 0, 1, 2]
        # bottom-left (row=1): min_col - 0 = 0, min_row - 1 = -1, max_col - 0 = 2, max_row - 1 = 1
        assert grid[1][0] == [0, -1, 2, 1]

    def test_missing_cells(self):
        cells = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A"},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D"},
        ]
        grid = cells_to_grid_top(cells)
        assert grid[0][0] == [0, 0, 1, 1]
        assert grid[0][1] == [0, 0, 1, 1]
        assert grid[1][0] == [0, 0, 1, 1]
        assert grid[1][1] == [0, 0, 1, 1]


class TestComputeFscore:
    def test_perfect_score(self):
        fscore, precision, recall = compute_fscore(10, 10, 10)
        assert fscore == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_no_predictions(self):
        fscore, precision, recall = compute_fscore(0, 10, 0)
        assert precision == 1.0
        assert recall == 0.0
        assert fscore == 0.0

    def test_no_true_instances(self):
        fscore, precision, recall = compute_fscore(0, 0, 10)
        assert precision == 0.0
        assert recall == 1.0
        assert fscore == 0.0

    def test_partial(self):
        fscore, precision, recall = compute_fscore(5, 10, 10)
        assert precision == 0.5
        assert recall == 0.5
        assert fscore == 0.5


class TestLcsSimilarity:
    def test_identical_strings(self):
        assert lcs_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        assert lcs_similarity("abc", "xyz") == 0.0

    def test_partial_overlap(self):
        assert lcs_similarity("abcde", "ace") == 0.75

    def test_symmetry(self):
        assert lcs_similarity("abcde", "bd") == lcs_similarity("bd", "abcde")


class TestGritsCon:
    def test_different_shapes(self):
        """Test main return path with different-shaped grids (bypasses early returns)."""
        true_grid = [["A"]]
        pred_grid = [["A", "X"]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_con(
            true_grid, pred_grid
        )
        assert isinstance(true_grid_scores, np.ndarray)
        assert isinstance(pred_grid_scores, np.ndarray)
        assert true_grid_scores.shape == (1, 1)
        assert pred_grid_scores.shape == (1, 2)
        assert total_match_score == 1.0
        assert true_grid_scores[0, 0] == 1.0
        assert pred_grid_scores[0, 0] == 1.0
        assert pred_grid_scores[0, 1] == 0.0

    def test_no_overlap(self):
        """Test grids with completely different text and different shapes."""
        true_grid = [["A", "B"]]
        pred_grid = [["X"], ["Y"]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_con(
            true_grid, pred_grid
        )
        assert total_match_score == 0.0
        assert true_grid_scores.sum() == 0.0
        assert pred_grid_scores.sum() == 0.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [["A", "B"], ["C", "D"]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_con(grid, grid)
        assert total_match_score == 4.0
        assert upper_bound == 4.0
        assert isinstance(true_grid_scores, np.ndarray)
        assert true_grid_scores.shape == (2, 2)


class TestGritsTop:
    def test_different_shapes(self):
        """Test main return path with different-shaped grids (bypasses early returns)."""
        true_grid = [[[0, 0, 1, 1]]]
        pred_grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_top(
            true_grid, pred_grid
        )
        assert isinstance(true_grid_scores, np.ndarray)
        assert isinstance(pred_grid_scores, np.ndarray)
        assert true_grid_scores.shape == (1, 1)
        assert pred_grid_scores.shape == (1, 2)
        assert total_match_score == 1.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_top(grid, grid)
        assert total_match_score == 4.0
        assert upper_bound == 4.0
        assert isinstance(true_grid_scores, np.ndarray)
        assert true_grid_scores.shape == (2, 2)


class TestHtmlToCells:
    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        cells = html_to_cells(html)
        assert len(cells) == 4
        assert cells[0]["cell_text"] == "A"
        assert cells[0]["row_nums"] == [0]
        assert cells[0]["column_nums"] == [0]

    def test_colspan(self):
        html = '<table><tr><td colspan="2">AB</td></tr><tr><td>C</td><td>D</td></tr></table>'
        cells = html_to_cells(html)
        assert len(cells) == 3
        assert cells[0]["column_nums"] == [0, 1]

    def test_rowspan(self):
        html = '<table><tr><td rowspan="2">AC</td><td>B</td></tr><tr><td>D</td></tr></table>'
        cells = html_to_cells(html)
        assert cells[0]["row_nums"] == [0, 1]

    def test_header_cell(self):
        html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"
        cells = html_to_cells(html)
        assert cells[0]["is_column_header"] is True
        assert cells[1]["is_column_header"] is False

    def test_invalid_html_returns_none(self):
        assert html_to_cells("not valid xml at all <<<") is None


class TestIou:
    def test_identical_boxes(self):
        bbox = BBox(0, 0, 10, 10)
        assert iou((bbox, bbox.area), (bbox, bbox.area)) == 1.0

    def test_no_overlap(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(6, 6, 10, 10)
        assert iou((a, a.area), (b, b.area)) == 0.0

    def test_partial_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        # intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
        assert iou((a, a.area), (b, b.area)) == pytest.approx(25.0 / 175.0)


class TestGritsLoc:
    def test_different_shapes(self):
        true_grid = [[[0, 0, 10, 10]]]
        pred_grid = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_loc(
            true_grid, pred_grid
        )
        assert isinstance(true_grid_scores, np.ndarray)
        assert true_grid_scores.shape == (1, 1)
        assert pred_grid_scores.shape == (1, 2)
        assert total_match_score == 1.0

    def test_identical_grids(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        total_match_score, upper_bound, true_grid_scores, pred_grid_scores = grits_loc(grid, grid)
        assert total_match_score == 4.0
        assert upper_bound == 4.0


class TestHungarianGritsLoc:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        tp_score, upper_bound, true_grid_scores, pred_grid_scores = hungarian_grits_loc(
            [grid], [grid]
        )
        assert tp_score == 4.0
        assert upper_bound == 4.0

    def test_multiple_grids(self):
        grid_a = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        grid_b = [[[50, 50, 60, 60], [60, 50, 70, 60]]]
        tp_score, _, _, _ = hungarian_grits_loc([grid_a, grid_b], [grid_b, grid_a])
        # Hungarian matching should pair them optimally: a-a and b-b
        assert tp_score == 4.0


class TestHungarianGritsCon:
    def test_single_grid_pair(self):
        true_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        pred_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        tp_score, upper_bound, true_grid_scores, pred_grid_scores = hungarian_grits_con(
            true_grids, pred_grids
        )
        assert tp_score == 4.0

    def test_multiple_grids(self):
        grid_a = np.array([["A", "B"]], dtype=object)
        grid_b = np.array([["X", "Y"]], dtype=object)
        true_grids = [grid_a, grid_b]
        pred_grids = [grid_b, grid_a]
        tp_score, _, _, _ = hungarian_grits_con(true_grids, pred_grids)
        # Hungarian matching should pair them optimally: a-a and b-b
        assert tp_score == 4.0


class TestHungarianGritsTop:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        tp_score, _, _, _ = hungarian_grits_top([grid], [grid])
        assert tp_score == 2.0


class TestGritsEvaluator:
    def test_perfect_score(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"], ["C", "D"]], dtype=object)
        evaluator.eval_sample(true_grid, true_grid)
        f1, precision, recall = evaluator.compute_grits()
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_multiple_adds(self):
        evaluator = GritsEvaluator(metric="con")
        grid1 = np.array([["A", "B"]], dtype=object)
        grid2 = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(grid1, grid1)
        evaluator.eval_sample(grid2, grid2)
        f1, precision, recall = evaluator.compute_grits()
        assert f1 == 1.0

    def test_compute_avg_grits(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        avg_f1, avg_precision, avg_recall = evaluator.compute_avg_grits()
        assert avg_f1 == 1.0
        assert avg_precision == 1.0
        assert avg_recall == 1.0

    def test_compute_avg_grits_precision_recall_order(self):
        evaluator = GritsEvaluator(metric="con")
        # true: 1x2, pred: 1x3 — recall should be 1.0, precision should be 2/3
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["A", "B", "X"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        avg_f1, avg_precision, avg_recall = evaluator.compute_avg_grits()
        assert avg_precision == pytest.approx(2.0 / 3.0)
        assert avg_recall == 1.0
        assert avg_f1 == pytest.approx(0.8)

    def test_accepts_single_numpy_array(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A"]], dtype=object)
        # Should not raise — eval_sample() wraps single arrays in a list
        evaluator.eval_sample(grid, grid)
        f1, _, _ = evaluator.compute_grits()
        assert f1 == 1.0

    def test_metric_top(self):
        evaluator = GritsEvaluator(metric="top")
        grid = np.array([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        evaluator.eval_sample(grid, grid)
        f1, _, _ = evaluator.compute_grits()
        assert f1 == 1.0

    def test_metric_loc(self):
        evaluator = GritsEvaluator(metric="loc")
        grid = np.array([[[0, 0, 10, 10], [10, 0, 20, 10]]])
        evaluator.eval_sample(grid, grid)
        f1, _, _ = evaluator.compute_grits()
        assert f1 == 1.0

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            GritsEvaluator(metric="invalid")
