# Copyright 2025-present Kensho Technologies, LLC.
import numpy as np
import pytest

from grits.bbox import BBox
from grits.conversion import (
    TableCell,
    cell_list_to_grid_con,
    cell_list_to_grid_loc,
    cell_list_to_grid_top,
    html_to_cell_list,
    html_to_grids,
)
from grits.grits import (
    GritsEvaluator,
    _GritsMetricEvaluator,
    compute_fscore,
    grits_con,
    grits_con_matching,
    grits_loc,
    grits_loc_matching,
    grits_top,
    grits_top_matching,
    hungarian_grits_con,
    hungarian_grits_con_matching,
    hungarian_grits_loc,
    hungarian_grits_loc_matching,
    hungarian_grits_top,
    hungarian_grits_top_matching,
    iou,
    lcs_similarity,
)


def _make_simple_cell_list():
    """Return a simple 2x2 table as a list of TableCells."""
    return [
        TableCell(row_nums=[0], column_nums=[0], cell_text="A", bbox=[0, 0, 10, 10]),
        TableCell(row_nums=[0], column_nums=[1], cell_text="B", bbox=[10, 0, 20, 10]),
        TableCell(row_nums=[1], column_nums=[0], cell_text="C", bbox=[0, 10, 10, 20]),
        TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[10, 10, 20, 20]),
    ]


class TestCellsToGridCon:
    def test_empty_cells(self):
        assert cell_list_to_grid_con([]) == [[]]

    def test_simple_text_grid(self):
        cell_list = _make_simple_cell_list()
        grid = cell_list_to_grid_con(cell_list)
        assert grid == [["A", "B"], ["C", "D"]]

    def test_spanning_cell(self):
        cell_list = [
            TableCell(row_nums=[0, 1], column_nums=[0], cell_text="span", bbox=[0, 0, 10, 20]),
            TableCell(row_nums=[0], column_nums=[1], cell_text="B", bbox=[10, 0, 20, 10]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[10, 10, 20, 20]),
        ]
        grid = cell_list_to_grid_con(cell_list)
        assert grid == [["span", "B"], ["span", "D"]]

    def test_missing_cells(self):
        cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="A", bbox=[0, 0, 10, 10]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[10, 10, 20, 20]),
        ]
        grid = cell_list_to_grid_con(cell_list)
        assert grid == [["A", ""], ["", "D"]]


class TestCellsToGridLoc:
    def test_empty_cells(self):
        assert cell_list_to_grid_loc([]) == [[]]

    def test_simple_bbox_grid(self):
        cell_list = _make_simple_cell_list()
        grid = cell_list_to_grid_loc(cell_list)
        assert grid == [
            [[0, 0, 10, 10], [10, 0, 20, 10]],
            [[0, 10, 10, 20], [10, 10, 20, 20]],
        ]

    def test_missing_cells(self):
        cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="A", bbox=[0, 0, 10, 10]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[10, 10, 20, 20]),
        ]
        grid = cell_list_to_grid_loc(cell_list)
        assert grid[0][0] == [0, 0, 10, 10]
        assert grid[0][1] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][0] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][1] == [10, 10, 20, 20]


class TestCellsToGridTop:
    def test_empty_cells(self):
        assert cell_list_to_grid_top([]) == [[]]

    def test_non_spanning(self):
        cell_list = _make_simple_cell_list()
        grid = cell_list_to_grid_top(cell_list)
        assert grid == [
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
        ]

    def test_spanning_cell(self):
        cell_list = [
            TableCell(row_nums=[0, 1], column_nums=[0, 1], cell_text="big"),
        ]
        grid = cell_list_to_grid_top(cell_list)
        # top-left: min_col=0, min_row=0, max_col=2, max_row=2
        assert grid[0][0] == [0, 0, 2, 2]
        # top-right (col=1): min_col - 1 = -1, min_row - 0 = 0, max_col - 1 = 1, max_row - 0 = 2
        assert grid[0][1] == [-1, 0, 1, 2]
        # bottom-left (row=1): min_col - 0 = 0, min_row - 1 = -1, max_col - 0 = 2, max_row - 1 = 1
        assert grid[1][0] == [0, -1, 2, 1]

    def test_missing_cells(self):
        cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="A"),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D"),
        ]
        grid = cell_list_to_grid_top(cell_list)
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
        result = grits_con_matching(true_grid, pred_grid)
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert isinstance(result.pred_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (1, 1)
        assert result.pred_grid_scores.shape == (1, 2)
        assert result.true_positive_score == 1.0
        assert result.true_grid_scores[0, 0] == 1.0
        assert result.pred_grid_scores[0, 0] == 1.0
        assert result.pred_grid_scores[0, 1] == 0.0

    def test_no_overlap(self):
        """Test grids with completely different text and different shapes."""
        true_grid = [["A", "B"]]
        pred_grid = [["X"], ["Y"]]
        result = grits_con_matching(true_grid, pred_grid)
        assert result.true_positive_score == 0.0
        assert result.true_grid_scores.sum() == 0.0
        assert result.pred_grid_scores.sum() == 0.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [["A", "B"], ["C", "D"]]
        result = grits_con_matching(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (2, 2)


class TestGritsConFscore:
    def test_identical_grids(self):
        grid = [["A", "B"], ["C", "D"]]
        f1, precision, recall = grits_con(grid, grid)
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_different_shapes(self):
        true_grid = [["A"]]
        pred_grid = [["A", "X"]]
        f1, precision, recall = grits_con(true_grid, pred_grid)
        assert recall == 1.0
        assert precision == 0.5
        assert f1 == pytest.approx(2 / 3)

    def test_no_overlap(self):
        true_grid = [["A", "B"]]
        pred_grid = [["X"], ["Y"]]
        f1, precision, recall = grits_con(true_grid, pred_grid)
        assert f1 == 0.0


class TestGritsTopFscore:
    def test_identical_grids(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        f1, precision, recall = grits_top(grid, grid)
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_different_shapes(self):
        true_grid = [[[0, 0, 1, 1]]]
        pred_grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        f1, precision, recall = grits_top(true_grid, pred_grid)
        assert recall == 1.0
        assert precision == 0.5
        assert f1 == pytest.approx(2 / 3)


class TestGritsLocFscore:
    def test_identical_grids(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        f1, precision, recall = grits_loc(grid, grid)
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_different_shapes(self):
        true_grid = [[[0, 0, 10, 10]]]
        pred_grid = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        f1, precision, recall = grits_loc(true_grid, pred_grid)
        assert recall == 1.0
        assert precision == 0.5
        assert f1 == pytest.approx(2 / 3)


class TestGritsTop:
    def test_different_shapes(self):
        """Test main return path with different-shaped grids (bypasses early returns)."""
        true_grid = [[[0, 0, 1, 1]]]
        pred_grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = grits_top_matching(true_grid, pred_grid)
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert isinstance(result.pred_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (1, 1)
        assert result.pred_grid_scores.shape == (1, 2)
        assert result.true_positive_score == 1.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = grits_top_matching(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (2, 2)


class TestHtmlToCellList:
    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        cell_list = html_to_cell_list(html)
        assert len(cell_list) == 4
        assert cell_list[0].cell_text == "A"
        assert cell_list[0].row_nums == [0]
        assert cell_list[0].column_nums == [0]

    def test_colspan(self):
        html = '<table><tr><td colspan="2">AB</td></tr><tr><td>C</td><td>D</td></tr></table>'
        cell_list = html_to_cell_list(html)
        assert len(cell_list) == 3
        assert cell_list[0].column_nums == [0, 1]

    def test_rowspan(self):
        html = '<table><tr><td rowspan="2">AC</td><td>B</td></tr><tr><td>D</td></tr></table>'
        cell_list = html_to_cell_list(html)
        assert cell_list[0].row_nums == [0, 1]

    def test_header_cell(self):
        html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"
        cell_list = html_to_cell_list(html)
        assert cell_list[0].is_column_header is True
        assert cell_list[1].is_column_header is False

    def test_invalid_html_returns_none(self):
        assert html_to_cell_list("not valid xml at all <<<") is None


class TestHtmlToGrids:
    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        result = html_to_grids(html)
        assert result is not None
        assert set(result.keys()) == {"con", "top"}
        assert result["con"] == [["A", "B"], ["C", "D"]]
        assert result["top"] == [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]

    def test_matches_individual_conversions(self):
        html = '<table><tr><td colspan="2">AB</td></tr><tr><td>C</td><td>D</td></tr></table>'
        cell_list = html_to_cell_list(html)
        result = html_to_grids(html)
        assert result is not None
        assert result["con"] == cell_list_to_grid_con(cell_list)
        assert result["top"] == cell_list_to_grid_top(cell_list)

    def test_invalid_html_returns_none(self):
        assert html_to_grids("not valid xml at all <<<") is None


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
        result = grits_loc_matching(true_grid, pred_grid)
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (1, 1)
        assert result.pred_grid_scores.shape == (1, 2)
        assert result.true_positive_score == 1.0

    def test_identical_grids(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        result = grits_loc_matching(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0


class TestHungarianGritsLoc:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        result = hungarian_grits_loc_matching([grid], [grid])
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0

    def test_multiple_grids(self):
        grid_a = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        grid_b = [[[50, 50, 60, 60], [60, 50, 70, 60]]]
        result = hungarian_grits_loc_matching([grid_a, grid_b], [grid_b, grid_a])
        # Hungarian matching should pair them optimally: a-a and b-b
        assert result.true_positive_score == 4.0


class TestHungarianGritsCon:
    def test_single_grid_pair(self):
        true_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        pred_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        result = hungarian_grits_con_matching(true_grids, pred_grids)
        assert result.true_positive_score == 4.0

    def test_multiple_grids(self):
        grid_a = np.array([["A", "B"]], dtype=object)
        grid_b = np.array([["X", "Y"]], dtype=object)
        true_grids = [grid_a, grid_b]
        pred_grids = [grid_b, grid_a]
        result = hungarian_grits_con_matching(true_grids, pred_grids)
        # Hungarian matching should pair them optimally: a-a and b-b
        assert result.true_positive_score == 4.0


class TestHungarianGritsLocFscore:
    def test_perfect_single_grid(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        f1, precision, recall = hungarian_grits_loc([grid], [grid])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_perfect_multiple_grids(self):
        grid_a = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        grid_b = [[[50, 50, 60, 60], [60, 50, 70, 60]]]
        f1, precision, recall = hungarian_grits_loc([grid_a, grid_b], [grid_b, grid_a])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_no_predictions(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        f1, precision, recall = hungarian_grits_loc([grid], [])
        assert f1 == 0.0
        assert precision == 1.0
        assert recall == 0.0

    def test_no_ground_truth(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        f1, precision, recall = hungarian_grits_loc([], [grid])
        assert f1 == 0.0
        assert precision == 0.0
        assert recall == 1.0


class TestHungarianGritsConFscore:
    def test_perfect_single_grid(self):
        grid = [["A", "B"], ["C", "D"]]
        f1, precision, recall = hungarian_grits_con([grid], [grid])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_perfect_multiple_grids(self):
        grid_a = [["A", "B"]]
        grid_b = [["X", "Y"]]
        f1, precision, recall = hungarian_grits_con([grid_a, grid_b], [grid_b, grid_a])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_no_predictions(self):
        grid = [["A", "B"]]
        f1, precision, recall = hungarian_grits_con([grid], [])
        assert f1 == 0.0
        assert precision == 1.0
        assert recall == 0.0

    def test_no_ground_truth(self):
        grid = [["A", "B"]]
        f1, precision, recall = hungarian_grits_con([], [grid])
        assert f1 == 0.0
        assert precision == 0.0
        assert recall == 1.0


class TestHungarianGritsTop:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = hungarian_grits_top_matching([grid], [grid])
        assert result.true_positive_score == 2.0


class TestHungarianGritsTopFscore:
    def test_perfect_single_grid(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        f1, precision, recall = hungarian_grits_top([grid], [grid])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_perfect_multiple_grids(self):
        grid1 = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        grid2 = [[[0, 0, 1, 1]]]
        f1, precision, recall = hungarian_grits_top([grid1, grid2], [grid1, grid2])
        assert f1 == 1.0
        assert precision == 1.0
        assert recall == 1.0

    def test_no_predictions(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        f1, precision, recall = hungarian_grits_top([grid], [])
        assert f1 == 0.0
        assert precision == 1.0
        assert recall == 0.0

    def test_no_ground_truth(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        f1, precision, recall = hungarian_grits_top([], [grid])
        assert f1 == 0.0
        assert precision == 0.0
        assert recall == 1.0


class TestGritsMetricEvaluator:
    def test_perfect_score(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"], ["C", "D"]]
        evaluator.eval_grids([true_grid], [true_grid])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_con_precision"] == 1.0
        assert results["grits_con_recall"] == 1.0

    def test_multiple_adds(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid1 = [["A", "B"]]
        grid2 = [["X", "Y"]]
        evaluator.eval_grids([grid1], [grid1])
        evaluator.eval_grids([grid2], [grid2])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0

    def test_compute_mean_grits_per_sample(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        evaluator.eval_grids([grid], [grid])
        results = evaluator.compute_mean_grits_per_sample()
        assert results["mean_grits_con_per_sample"] == 1.0
        assert results["mean_grits_con_precision_per_sample"] == 1.0
        assert results["mean_grits_con_recall_per_sample"] == 1.0

    def test_compute_mean_grits_per_sample_precision_recall_order(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        # true: 1x2, pred: 1x3 — recall should be 1.0, precision should be 2/3
        true_grid = [["A", "B"]]
        pred_grid = [["A", "B", "X"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        results = evaluator.compute_mean_grits_per_sample()
        assert results["mean_grits_con_precision_per_sample"] == pytest.approx(2.0 / 3.0)
        assert results["mean_grits_con_recall_per_sample"] == 1.0
        assert results["mean_grits_con_per_sample"] == pytest.approx(0.8)

    def test_metric_top(self):
        evaluator = _GritsMetricEvaluator(metric="top")
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        evaluator.eval_grids([grid], [grid])
        assert evaluator.compute_grits()["grits_top"] == 1.0

    def test_metric_loc(self):
        evaluator = _GritsMetricEvaluator(metric="loc")
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        evaluator.eval_grids([grid], [grid])
        assert evaluator.compute_grits()["grits_loc"] == 1.0

    def test_tracks_grid_counts(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid1 = [["A", "B"]]
        grid2 = [["X", "Y"]]
        evaluator.eval_grids([grid1, grid2], [grid1])
        assert evaluator.num_true_grids == [2]
        assert evaluator.num_pred_grids == [1]
        evaluator.eval_grids([grid1], [grid1, grid2])
        assert evaluator.num_true_grids == [2, 1]
        assert evaluator.num_pred_grids == [1, 2]

    def test_exact_match_accuracy_perfect(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        evaluator.eval_grids([grid], [grid])
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 1.0

    def test_exact_match_accuracy_no_match(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"]]
        pred_grid = [["X", "Y"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 0.0

    def test_exact_match_accuracy_mixed(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        diff_grid = [["X", "Y"]]
        evaluator.eval_grids([grid], [grid])  # exact match
        evaluator.eval_grids([grid], [diff_grid])  # not exact
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 0.5

    def test_avg_exact_match_accuracy_perfect(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        evaluator.eval_grids([grid], [grid])
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_grid_exact_match_accuracy_per_sample"
            ]
            == 1.0
        )

    def test_avg_exact_match_accuracy_no_match(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"]]
        pred_grid = [["X", "Y"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_grid_exact_match_accuracy_per_sample"
            ]
            == 0.0
        )

    def test_avg_exact_match_accuracy_mixed(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        diff_grid = [["X", "Y"]]
        evaluator.eval_grids([grid], [grid])  # exact match
        evaluator.eval_grids([grid], [diff_grid])  # not exact
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_grid_exact_match_accuracy_per_sample"
            ]
            == 0.5
        )

    def test_cell_exact_match_accuracy_perfect(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        evaluator.eval_grids([grid], [grid])
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 1.0

    def test_cell_exact_match_accuracy_no_match(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"]]
        pred_grid = [["X", "Y"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 0.0

    def test_cell_exact_match_accuracy_mixed(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"], ["C", "D"]]
        diff_grid = [["A", "X"], ["C", "D"]]
        evaluator.eval_grids([grid], [diff_grid])
        # 3 of 4 true cells match exactly
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 0.75

    def test_avg_cell_exact_match_accuracy_perfect(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"]]
        evaluator.eval_grids([grid], [grid])
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_cell_exact_match_accuracy_per_sample"
            ]
            == 1.0
        )

    def test_avg_cell_exact_match_accuracy_no_match(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"]]
        pred_grid = [["X", "Y"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_cell_exact_match_accuracy_per_sample"
            ]
            == 0.0
        )

    def test_avg_cell_exact_match_accuracy_mixed(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid_a = [["A", "B"]]
        grid_b = [["A", "X"]]
        evaluator.eval_grids([grid_a], [grid_a])  # 1.0 cell accuracy
        evaluator.eval_grids([grid_a], [grid_b])  # 0.5 cell accuracy
        assert (
            evaluator.compute_mean_grits_per_sample()[
                "mean_grits_con_cell_exact_match_accuracy_per_sample"
            ]
            == 0.75
        )

    def test_exact_cell_matches_perfect(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        grid = [["A", "B"], ["C", "D"]]
        evaluator.eval_grids([grid], [grid])
        assert evaluator.num_exact_cell_matches == [4]

    def test_exact_cell_matches_partial(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"], ["C", "D"]]
        pred_grid = [["A", "X"], ["C", "D"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        # "A", "C", "D" match exactly; "B" vs "X" does not
        assert evaluator.num_exact_cell_matches == [3]

    def test_exact_cell_matches_no_match(self):
        evaluator = _GritsMetricEvaluator(metric="con")
        true_grid = [["A", "B"]]
        pred_grid = [["X", "Y"]]
        evaluator.eval_grids([true_grid], [pred_grid])
        assert evaluator.num_exact_cell_matches == [0]

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            _GritsMetricEvaluator(metric="invalid")


class TestGritsEvaluator:
    def test_default_metrics(self):
        evaluator = GritsEvaluator()
        assert evaluator.metrics == ["con", "top"]

    def test_custom_metrics(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        assert evaluator.metrics == ["con", "top"]

    def test_eval_grids_by_metric_perfect_score(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        true_grid_con = [["A", "B"], ["C", "D"]]
        true_grid_top = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        true_grids = {"con": [true_grid_con], "top": [true_grid_top]}
        evaluator.eval_grids_by_metric(true_grids, true_grids)
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0
            assert results[f"grits_{metric}_precision"] == 1.0
            assert results[f"grits_{metric}_recall"] == 1.0

    def test_compute_mean_grits_per_sample(self):
        evaluator = GritsEvaluator(metrics=["con"])
        grid = [["A", "B"]]
        evaluator.eval_grids_by_metric({"con": [grid]}, {"con": [grid]})
        results = evaluator.compute_mean_grits_per_sample()
        assert results["mean_grits_con_per_sample"] == 1.0
        assert results["mean_grits_con_precision_per_sample"] == 1.0
        assert results["mean_grits_con_recall_per_sample"] == 1.0

    def test_eval_htmls_perfect_score(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([html], [html])
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0
            assert results[f"grits_{metric}_precision"] == 1.0
            assert results[f"grits_{metric}_recall"] == 1.0

    def test_eval_htmls_different_tables(self):
        true_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        pred_html = "<table><tr><td>A</td><td>X</td></tr></table>"
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([true_html], [pred_html])
        results = evaluator.compute_grits()
        # "top" should be perfect (same structure), "con" should not (different text)
        assert results["grits_top"] == 1.0
        assert results["grits_con"] == 0.5

    def test_eval_htmls_invalid_html_raises(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        with pytest.raises(ValueError, match="Failed to parse true_html"):
            evaluator.eval_htmls(["not valid html<><<"], ["<table><tr><td>A</td></tr></table>"])
        with pytest.raises(ValueError, match="Failed to parse pred_html"):
            evaluator.eval_htmls(["<table><tr><td>A</td></tr></table>"], ["not valid html<><<"])

    def test_eval_htmls_list_perfect_score(self):
        html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
        html_b = "<table><tr><td>X</td><td>Y</td></tr></table>"
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([html_a, html_b], [html_a, html_b])
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_htmls_list_hungarian_matching(self):
        html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
        html_b = "<table><tr><td>X</td><td>Y</td></tr></table>"
        evaluator = GritsEvaluator(metrics=["con", "top"])
        # Swapped order — Hungarian matching should pair them optimally
        evaluator.eval_htmls([html_a, html_b], [html_b, html_a])
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_htmls_list_invalid_raises(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        with pytest.raises(ValueError, match="Failed to parse true_html"):
            evaluator.eval_htmls(
                ["<table><tr><td>A</td></tr></table>", "not valid<<>"],
                ["<table><tr><td>A</td></tr></table>"],
            )
        with pytest.raises(ValueError, match="Failed to parse pred_html"):
            evaluator.eval_htmls(
                ["<table><tr><td>A</td></tr></table>"],
                ["not valid<<>"],
            )

    def test_eval_htmls_loc_metric_raises(self):
        evaluator = GritsEvaluator(metrics=["con", "loc"])
        with pytest.raises(ValueError, match="eval_htmls cannot evaluate the 'loc' metric"):
            evaluator.eval_htmls(
                ["<table><tr><td>A</td></tr></table>"], ["<table><tr><td>A</td></tr></table>"]
            )

    def test_eval_htmls_single_metric(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        evaluator = GritsEvaluator(metrics=["con"])
        evaluator.eval_htmls([html], [html])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert "grits_top" not in results

    def test_omits_metrics_without_samples(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        html = "<table><tr><td>A</td></tr></table>"
        evaluator.eval_htmls([html], [html])
        results = evaluator.compute_grits()
        assert "grits_con" in results
        assert "grits_top" in results
        assert "grits_loc" not in results
        avg_results = evaluator.compute_mean_grits_per_sample()
        assert "mean_grits_con_per_sample" in avg_results
        assert "mean_grits_top_per_sample" in avg_results
        assert "mean_grits_loc_per_sample" not in avg_results

    def test_eval_table_cell_lists_perfect(self):
        evaluator = GritsEvaluator(metrics=["con", "loc", "top"])
        cell_list = _make_simple_cell_list()
        evaluator.eval_table_cell_lists([cell_list], [cell_list])
        results = evaluator.compute_grits()
        for metric in ("con", "loc", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_table_cell_lists_different_text(self):
        evaluator = GritsEvaluator(metrics=["con", "loc", "top"])
        true_cell_list = _make_simple_cell_list()
        pred_cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="X", bbox=[0, 0, 10, 10]),
            TableCell(row_nums=[0], column_nums=[1], cell_text="Y", bbox=[10, 0, 20, 10]),
            TableCell(row_nums=[1], column_nums=[0], cell_text="Z", bbox=[0, 10, 10, 20]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="W", bbox=[10, 10, 20, 20]),
        ]
        evaluator.eval_table_cell_lists([true_cell_list], [pred_cell_list])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 0.0
        assert results["grits_loc"] == 1.0
        assert results["grits_top"] == 1.0

    def test_eval_table_cell_lists_list(self):
        evaluator = GritsEvaluator(metrics=["con", "loc", "top"])
        cell_list_a = _make_simple_cell_list()
        cell_list_b = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="X", bbox=[0, 0, 5, 5]),
        ]
        evaluator.eval_table_cell_lists([cell_list_a, cell_list_b], [cell_list_a, cell_list_b])
        results = evaluator.compute_grits()
        for metric in ("con", "loc", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_table_cell_lists_different_bbox(self):
        evaluator = GritsEvaluator(metrics=["con", "loc", "top"])
        true_cell_list = _make_simple_cell_list()
        # Same text and structure, but shifted bboxes
        pred_cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="A", bbox=[1, 1, 10, 10]),
            TableCell(row_nums=[0], column_nums=[1], cell_text="B", bbox=[11, 1, 20, 10]),
            TableCell(row_nums=[1], column_nums=[0], cell_text="C", bbox=[1, 11, 10, 20]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[11, 11, 20, 20]),
        ]
        evaluator.eval_table_cell_lists([true_cell_list], [pred_cell_list])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_loc"] == pytest.approx(0.81, abs=1e-4)
        assert results["grits_top"] == 1.0

    def test_eval_table_cell_lists_different_structure(self):
        evaluator = GritsEvaluator(metrics=["con", "loc", "top"])
        true_cell_list = _make_simple_cell_list()
        # Same text and bboxes, but one cell spans two columns
        pred_cell_list = [
            TableCell(row_nums=[0], column_nums=[0, 1], cell_text="A", bbox=[0, 0, 10, 10]),
            TableCell(row_nums=[1], column_nums=[0], cell_text="C", bbox=[0, 10, 10, 20]),
            TableCell(row_nums=[1], column_nums=[1], cell_text="D", bbox=[10, 10, 20, 20]),
        ]
        evaluator.eval_table_cell_lists([true_cell_list], [pred_cell_list])
        results = evaluator.compute_grits()
        assert results["grits_top"] == 0.75

    def test_eval_table_cell_lists_subset_of_metrics(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        cell_list = _make_simple_cell_list()
        evaluator.eval_table_cell_lists([cell_list], [cell_list])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_top"] == 1.0
        assert "grits_loc" not in results

    def test_eval_grids_perfect_score(self):
        evaluator = GritsEvaluator(metrics=["con"])
        grid = [["A", "B"], ["C", "D"]]
        evaluator.eval_grids([grid], [grid])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_con_precision"] == 1.0
        assert results["grits_con_recall"] == 1.0

    def test_eval_grids_multiple_metrics_raises(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        grid = [["A", "B"]]
        with pytest.raises(
            TypeError, match="eval_grids requires the evaluator to track exactly one metric"
        ):
            evaluator.eval_grids([grid], [grid])

    def test_eval_htmls_bare_string_raises(self):
        evaluator = GritsEvaluator()
        html = "<table><tr><td>A</td></tr></table>"
        with pytest.raises(TypeError, match="eval_htmls expects lists"):
            evaluator.eval_htmls(html, html)

    def test_eval_table_cell_lists_single_list_raises(self):
        evaluator = GritsEvaluator()
        cell_list = [
            TableCell(row_nums=[0], column_nums=[0], cell_text="A", bbox=[0, 0, 1, 1]),
        ]
        with pytest.raises(TypeError, match="eval_table_cell_lists expects a list of lists"):
            evaluator.eval_table_cell_lists(cell_list, cell_list)

    def test_mismatched_sample_counts_raises(self):
        evaluator = GritsEvaluator(metrics=["con", "top"])
        con_grid = [["A", "B"]]
        top_grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        evaluator.eval_grids_by_metric(
            {"con": [con_grid], "top": [top_grid]}, {"con": [con_grid], "top": [top_grid]}
        )
        evaluator._evaluators["con"].eval_grids([con_grid], [con_grid])  # noqa: SLF001
        with pytest.raises(ValueError, match="Mismatched sample counts"):
            evaluator.compute_grits()

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            GritsEvaluator(metrics=["invalid"])


# HTML fixtures for integration tests
_TRUE_HTML_1 = (
    "<table><thead><tr><th>Shape</th><th>Color</th></tr></thead>"
    "<tbody><tr><td>Circle</td><td>Red</td></tr>"
    "<tr><td>Triangle</td><td>Blue</td></tr></tbody></table>"
)
_TRUE_HTML_2 = (
    "<table><tr><td>Week 1</td><td>1 km</td><td>6:02</td></tr>"
    "<tr><td>Week 2</td><td>1 km</td><td>5:25</td></tr>"
    "<tr><td>Week 3</td><td>2 km</td><td>10:19</td></tr></table>"
)
_PRED_HTML_1 = (
    "<table><thead><tr><th>Shape</th><th>Color</th></tr></thead>"
    "<tbody><tr><td>Cirle</td><td>Orange</td></tr>"
    "<tr><td>Triangle</td><td>Bleu</td></tr></tbody></table>"
)
_PRED_HTML_2 = (
    "<table><tr><td>Week I</td><td>1 m</td><td>6:02</td></tr>"
    "<tr><td>Week 2</td><td>1 kn</td><td>5:25</td></tr></table>"
)
_PRED_HTML_3 = "<table><tr><td>Week 3</td><td>2 km</td><td>10:19</td></tr></table>"


class TestHtmlIntegration:
    def test_single_table_imperfect_text(self):
        """Same structure, some text differences."""
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([_TRUE_HTML_1], [_PRED_HTML_1])
        results = evaluator.compute_grits()
        # Structure is identical
        assert results["grits_top"] == 1.0
        # Text has differences (Cirle, Orange, Bleu)
        assert results["grits_con"] == pytest.approx(0.8136, abs=1e-4)
        counts = evaluator.compute_counts()
        assert counts["num_true_cells"] == 6
        assert counts["num_pred_cells"] == 6

    def test_single_table_missing_row(self):
        """Prediction has 2 of 3 rows."""
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([_TRUE_HTML_2], [_PRED_HTML_2])
        results = evaluator.compute_grits()
        # Missing row lowers recall
        assert results["grits_top_recall"] == pytest.approx(2.0 / 3.0, abs=1e-4)
        assert results["grits_top_precision"] == 1.0
        assert results["grits_con"] == pytest.approx(0.7254, abs=1e-4)
        counts = evaluator.compute_counts()
        assert counts["num_true_cells"] == 9
        assert counts["num_pred_cells"] == 6

    def test_multi_table_hungarian_matching(self):
        """Two true tables matched against three predicted tables."""
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls(
            [_TRUE_HTML_1, _TRUE_HTML_2], [_PRED_HTML_1, _PRED_HTML_2, _PRED_HTML_3]
        )
        results = evaluator.compute_grits()
        # Top metric: pred tables split true_html_2, so not perfect
        assert results["grits_top"] == pytest.approx(0.8, abs=1e-4)
        assert results["grits_con"] == pytest.approx(0.6881, abs=1e-4)
        counts = evaluator.compute_counts()
        assert counts["num_true_tables"] == 2
        assert counts["num_pred_tables"] == 3

    def test_two_samples_micro_vs_macro(self):
        """Two separate samples, compare micro and macro averaging."""
        evaluator = GritsEvaluator(metrics=["con", "top"])
        evaluator.eval_htmls([_TRUE_HTML_1], [_PRED_HTML_1])
        evaluator.eval_htmls([_TRUE_HTML_2], [_PRED_HTML_2])
        micro = evaluator.compute_grits()
        macro = evaluator.compute_mean_grits_per_sample()
        counts = evaluator.compute_counts()
        # Micro and macro should differ because sample sizes differ (6 vs 9 cells)
        assert micro["grits_con"] != macro["mean_grits_con_per_sample"]
        assert micro["grits_con"] == pytest.approx(0.7646, abs=1e-4)
        assert macro["mean_grits_con_per_sample"] == pytest.approx(0.7695, abs=1e-4)
        assert counts["num_samples"] == 2
        assert counts["num_true_tables"] == 2
        assert counts["num_pred_tables"] == 2


class TestTableCell:
    def test_construction_all_fields(self):
        cell = TableCell(
            row_nums=[0, 1],
            column_nums=[0],
            cell_text="hello",
            bbox=[0.0, 0.0, 10.0, 20.0],
            is_column_header=True,
            is_row_header=True,
        )
        assert cell.row_nums == [0, 1]
        assert cell.column_nums == [0]
        assert cell.cell_text == "hello"
        assert cell.bbox == [0.0, 0.0, 10.0, 20.0]
        assert cell.is_column_header is True
        assert cell.is_row_header is True

    def test_construction_defaults(self):
        cell = TableCell(row_nums=[0], column_nums=[1])
        assert cell.cell_text == ""
        assert cell.bbox is None
        assert cell.is_column_header is False
        assert cell.is_row_header is False

    def test_from_dict_all_keys(self):
        d = {
            "row_nums": [0],
            "column_nums": [1],
            "cell_text": "A",
            "bbox": [0, 0, 10, 10],
            "is_column_header": True,
            "is_row_header": True,
        }
        cell = TableCell.from_dict(d)
        assert cell.row_nums == [0]
        assert cell.column_nums == [1]
        assert cell.cell_text == "A"
        assert cell.bbox == [0, 0, 10, 10]
        assert cell.is_column_header is True
        assert cell.is_row_header is True

    def test_from_dict_minimal_keys(self):
        d = {"row_nums": [0], "column_nums": [1]}
        cell = TableCell.from_dict(d)
        assert cell.cell_text == ""
        assert cell.bbox is None
        assert cell.is_column_header is False
        assert cell.is_row_header is False

    def test_from_dict_ignores_extra_keys(self):
        d = {"row_nums": [0], "column_nums": [1], "extra_field": 42, "another": "value"}
        cell = TableCell.from_dict(d)
        assert cell.row_nums == [0]
        assert cell.column_nums == [1]
        assert not hasattr(cell, "extra_field")

    def test_to_dict_with_bbox(self):
        cell = TableCell(
            row_nums=[0], column_nums=[1], cell_text="A", bbox=[0, 0, 10, 10], is_column_header=True
        )
        d = cell.to_dict()
        assert d == {
            "row_nums": [0],
            "column_nums": [1],
            "cell_text": "A",
            "bbox": [0, 0, 10, 10],
            "is_column_header": True,
            "is_row_header": False,
        }

    def test_to_dict_without_bbox(self):
        cell = TableCell(row_nums=[0], column_nums=[1], cell_text="A")
        d = cell.to_dict()
        assert d == {
            "row_nums": [0],
            "column_nums": [1],
            "cell_text": "A",
            "is_column_header": False,
            "is_row_header": False,
        }
        assert "bbox" not in d

    def test_round_trip(self):
        cell = TableCell(row_nums=[0, 1], column_nums=[2], cell_text="X", bbox=[1, 2, 3, 4])
        assert TableCell.from_dict(cell.to_dict()) == cell

    def test_frozen(self):
        cell = TableCell(row_nums=[0], column_nums=[0])
        with pytest.raises(AttributeError):
            cell.cell_text = "changed"
