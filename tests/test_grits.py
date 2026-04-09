# Copyright 2025-present Kensho Technologies, LLC.
import numpy as np
import pytest

from grits.bbox import BBox
from grits.grits import (
    GritsEvaluator,
    MultiGritsEvaluator,
    cell_dicts_to_grid_con,
    cell_dicts_to_grid_loc,
    cell_dicts_to_grid_top,
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
from grits.utils import html_to_cell_dicts


def _make_simple_cell_dicts():
    """Return a simple 2x2 table as a list of cell dicts."""
    return [
        {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
        {"row_nums": [0], "column_nums": [1], "cell_text": "B", "bbox": [10, 0, 20, 10]},
        {"row_nums": [1], "column_nums": [0], "cell_text": "C", "bbox": [0, 10, 10, 20]},
        {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
    ]


class TestCellsToGridCon:
    def test_empty_cells(self):
        assert cell_dicts_to_grid_con([]) == [[]]

    def test_simple_text_grid(self):
        cell_dicts = _make_simple_cell_dicts()
        grid = cell_dicts_to_grid_con(cell_dicts)
        assert grid == [["A", "B"], ["C", "D"]]

    def test_spanning_cell(self):
        cell_dicts = [
            {"row_nums": [0, 1], "column_nums": [0], "cell_text": "span", "bbox": [0, 0, 10, 20]},
            {"row_nums": [0], "column_nums": [1], "cell_text": "B", "bbox": [10, 0, 20, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cell_dicts_to_grid_con(cell_dicts)
        assert grid == [["span", "B"], ["span", "D"]]

    def test_missing_cells(self):
        cell_dicts = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cell_dicts_to_grid_con(cell_dicts)
        assert grid == [["A", ""], ["", "D"]]


class TestCellsToGridLoc:
    def test_empty_cells(self):
        assert cell_dicts_to_grid_loc([]) == [[]]

    def test_simple_bbox_grid(self):
        cell_dicts = _make_simple_cell_dicts()
        grid = cell_dicts_to_grid_loc(cell_dicts)
        assert grid == [
            [[0, 0, 10, 10], [10, 0, 20, 10]],
            [[0, 10, 10, 20], [10, 10, 20, 20]],
        ]

    def test_missing_cells(self):
        cell_dicts = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [0, 0, 10, 10]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        grid = cell_dicts_to_grid_loc(cell_dicts)
        assert grid[0][0] == [0, 0, 10, 10]
        assert grid[0][1] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][0] == [0.0, 0.0, 0.0, 0.0]
        assert grid[1][1] == [10, 10, 20, 20]


class TestCellsToGridTop:
    def test_empty_cells(self):
        assert cell_dicts_to_grid_top([]) == [[]]

    def test_non_spanning(self):
        cell_dicts = _make_simple_cell_dicts()
        grid = cell_dicts_to_grid_top(cell_dicts)
        assert grid == [
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
        ]

    def test_spanning_cell(self):
        cell_dicts = [
            {"row_nums": [0, 1], "column_nums": [0, 1], "cell_text": "big"},
        ]
        grid = cell_dicts_to_grid_top(cell_dicts)
        # top-left: min_col=0, min_row=0, max_col=2, max_row=2
        assert grid[0][0] == [0, 0, 2, 2]
        # top-right (col=1): min_col - 1 = -1, min_row - 0 = 0, max_col - 1 = 1, max_row - 0 = 2
        assert grid[0][1] == [-1, 0, 1, 2]
        # bottom-left (row=1): min_col - 0 = 0, min_row - 1 = -1, max_col - 0 = 2, max_row - 1 = 1
        assert grid[1][0] == [0, -1, 2, 1]

    def test_missing_cells(self):
        cell_dicts = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A"},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D"},
        ]
        grid = cell_dicts_to_grid_top(cell_dicts)
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
        result = grits_con(true_grid, pred_grid)
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
        result = grits_con(true_grid, pred_grid)
        assert result.true_positive_score == 0.0
        assert result.true_grid_scores.sum() == 0.0
        assert result.pred_grid_scores.sum() == 0.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [["A", "B"], ["C", "D"]]
        result = grits_con(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (2, 2)


class TestGritsTop:
    def test_different_shapes(self):
        """Test main return path with different-shaped grids (bypasses early returns)."""
        true_grid = [[[0, 0, 1, 1]]]
        pred_grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = grits_top(true_grid, pred_grid)
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert isinstance(result.pred_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (1, 1)
        assert result.pred_grid_scores.shape == (1, 2)
        assert result.true_positive_score == 1.0

    def test_identical_grids(self):
        """Test early exit path when grids are identical."""
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = grits_top(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (2, 2)


class TestHtmlToCells:
    def test_simple_table(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        cell_dicts = html_to_cell_dicts(html)
        assert len(cell_dicts) == 4
        assert cell_dicts[0]["cell_text"] == "A"
        assert cell_dicts[0]["row_nums"] == [0]
        assert cell_dicts[0]["column_nums"] == [0]

    def test_colspan(self):
        html = '<table><tr><td colspan="2">AB</td></tr><tr><td>C</td><td>D</td></tr></table>'
        cell_dicts = html_to_cell_dicts(html)
        assert len(cell_dicts) == 3
        assert cell_dicts[0]["column_nums"] == [0, 1]

    def test_rowspan(self):
        html = '<table><tr><td rowspan="2">AC</td><td>B</td></tr><tr><td>D</td></tr></table>'
        cell_dicts = html_to_cell_dicts(html)
        assert cell_dicts[0]["row_nums"] == [0, 1]

    def test_header_cell(self):
        html = "<table><tr><th>Header</th></tr><tr><td>Data</td></tr></table>"
        cell_dicts = html_to_cell_dicts(html)
        assert cell_dicts[0]["is_column_header"] is True
        assert cell_dicts[1]["is_column_header"] is False

    def test_invalid_html_returns_none(self):
        assert html_to_cell_dicts("not valid xml at all <<<") is None


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
        result = grits_loc(true_grid, pred_grid)
        assert isinstance(result.true_grid_scores, np.ndarray)
        assert result.true_grid_scores.shape == (1, 1)
        assert result.pred_grid_scores.shape == (1, 2)
        assert result.true_positive_score == 1.0

    def test_identical_grids(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        result = grits_loc(grid, grid)
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0


class TestHungarianGritsLoc:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 10, 10], [10, 0, 20, 10]], [[0, 10, 10, 20], [10, 10, 20, 20]]]
        result = hungarian_grits_loc([grid], [grid])
        assert result.true_positive_score == 4.0
        assert result.true_positive_score_upper_bound == 4.0

    def test_multiple_grids(self):
        grid_a = [[[0, 0, 10, 10], [10, 0, 20, 10]]]
        grid_b = [[[50, 50, 60, 60], [60, 50, 70, 60]]]
        result = hungarian_grits_loc([grid_a, grid_b], [grid_b, grid_a])
        # Hungarian matching should pair them optimally: a-a and b-b
        assert result.true_positive_score == 4.0


class TestHungarianGritsCon:
    def test_single_grid_pair(self):
        true_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        pred_grids = [np.array([["A", "B"], ["C", "D"]], dtype=object)]
        result = hungarian_grits_con(true_grids, pred_grids)
        assert result.true_positive_score == 4.0

    def test_multiple_grids(self):
        grid_a = np.array([["A", "B"]], dtype=object)
        grid_b = np.array([["X", "Y"]], dtype=object)
        true_grids = [grid_a, grid_b]
        pred_grids = [grid_b, grid_a]
        result = hungarian_grits_con(true_grids, pred_grids)
        # Hungarian matching should pair them optimally: a-a and b-b
        assert result.true_positive_score == 4.0


class TestHungarianGritsTop:
    def test_single_grid_pair(self):
        grid = [[[0, 0, 1, 1], [0, 0, 1, 1]]]
        result = hungarian_grits_top([grid], [grid])
        assert result.true_positive_score == 2.0


class TestGritsEvaluator:
    def test_perfect_score(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"], ["C", "D"]], dtype=object)
        evaluator.eval_sample(true_grid, true_grid)
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_con_precision"] == 1.0
        assert results["grits_con_recall"] == 1.0

    def test_multiple_adds(self):
        evaluator = GritsEvaluator(metric="con")
        grid1 = np.array([["A", "B"]], dtype=object)
        grid2 = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(grid1, grid1)
        evaluator.eval_sample(grid2, grid2)
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0

    def test_compute_avg_grits(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        results = evaluator.compute_avg_grits()
        assert results["avg_grits_con"] == 1.0
        assert results["avg_grits_con_precision"] == 1.0
        assert results["avg_grits_con_recall"] == 1.0

    def test_compute_avg_grits_precision_recall_order(self):
        evaluator = GritsEvaluator(metric="con")
        # true: 1x2, pred: 1x3 — recall should be 1.0, precision should be 2/3
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["A", "B", "X"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        results = evaluator.compute_avg_grits()
        assert results["avg_grits_con_precision"] == pytest.approx(2.0 / 3.0)
        assert results["avg_grits_con_recall"] == 1.0
        assert results["avg_grits_con"] == pytest.approx(0.8)

    def test_accepts_single_numpy_array(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A"]], dtype=object)
        # Should not raise — eval_sample() wraps single arrays in a list
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_grits()["grits_con"] == 1.0

    def test_metric_top(self):
        evaluator = GritsEvaluator(metric="top")
        grid = np.array([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_grits()["grits_top"] == 1.0

    def test_metric_loc(self):
        evaluator = GritsEvaluator(metric="loc")
        grid = np.array([[[0, 0, 10, 10], [10, 0, 20, 10]]])
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_grits()["grits_loc"] == 1.0

    def test_tracks_grid_counts(self):
        evaluator = GritsEvaluator(metric="con")
        grid1 = np.array([["A", "B"]], dtype=object)
        grid2 = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample([grid1, grid2], [grid1])
        assert evaluator.num_true_grids == [2]
        assert evaluator.num_pred_grids == [1]
        evaluator.eval_sample([grid1], [grid1, grid2])
        assert evaluator.num_true_grids == [2, 1]
        assert evaluator.num_pred_grids == [1, 2]

    def test_grid_counts_single_numpy_array(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.num_true_grids == [1]
        assert evaluator.num_pred_grids == [1]

    def test_exact_match_accuracy_perfect(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 1.0

    def test_exact_match_accuracy_no_match(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 0.0

    def test_exact_match_accuracy_mixed(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        diff_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(grid, grid)  # exact match
        evaluator.eval_sample(grid, diff_grid)  # not exact
        assert evaluator.compute_grits()["grits_con_grid_exact_match_accuracy"] == 0.5

    def test_avg_exact_match_accuracy_perfect(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_avg_grits()["avg_grits_con_grid_exact_match_accuracy"] == 1.0

    def test_avg_exact_match_accuracy_no_match(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        assert evaluator.compute_avg_grits()["avg_grits_con_grid_exact_match_accuracy"] == 0.0

    def test_avg_exact_match_accuracy_mixed(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        diff_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(grid, grid)  # exact match
        evaluator.eval_sample(grid, diff_grid)  # not exact
        assert evaluator.compute_avg_grits()["avg_grits_con_grid_exact_match_accuracy"] == 0.5

    def test_cell_exact_match_accuracy_perfect(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 1.0

    def test_cell_exact_match_accuracy_no_match(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 0.0

    def test_cell_exact_match_accuracy_mixed(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"], ["C", "D"]], dtype=object)
        diff_grid = np.array([["A", "X"], ["C", "D"]], dtype=object)
        evaluator.eval_sample(grid, diff_grid)
        # 3 of 4 true cells match exactly
        assert evaluator.compute_grits()["grits_con_cell_exact_match_accuracy"] == 0.75

    def test_avg_cell_exact_match_accuracy_perfect(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.compute_avg_grits()["avg_grits_con_cell_exact_match_accuracy"] == 1.0

    def test_avg_cell_exact_match_accuracy_no_match(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        assert evaluator.compute_avg_grits()["avg_grits_con_cell_exact_match_accuracy"] == 0.0

    def test_avg_cell_exact_match_accuracy_mixed(self):
        evaluator = GritsEvaluator(metric="con")
        grid_a = np.array([["A", "B"]], dtype=object)
        grid_b = np.array([["A", "X"]], dtype=object)
        evaluator.eval_sample(grid_a, grid_a)  # 1.0 cell accuracy
        evaluator.eval_sample(grid_a, grid_b)  # 0.5 cell accuracy
        assert evaluator.compute_avg_grits()["avg_grits_con_cell_exact_match_accuracy"] == 0.75

    def test_exact_cell_matches_perfect(self):
        evaluator = GritsEvaluator(metric="con")
        grid = np.array([["A", "B"], ["C", "D"]], dtype=object)
        evaluator.eval_sample(grid, grid)
        assert evaluator.num_exact_cell_matches == [4]

    def test_exact_cell_matches_partial(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"], ["C", "D"]], dtype=object)
        pred_grid = np.array([["A", "X"], ["C", "D"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        # "A", "C", "D" match exactly; "B" vs "X" does not
        assert evaluator.num_exact_cell_matches == [3]

    def test_exact_cell_matches_no_match(self):
        evaluator = GritsEvaluator(metric="con")
        true_grid = np.array([["A", "B"]], dtype=object)
        pred_grid = np.array([["X", "Y"]], dtype=object)
        evaluator.eval_sample(true_grid, pred_grid)
        assert evaluator.num_exact_cell_matches == [0]

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            GritsEvaluator(metric="invalid")


class TestMultiGritsEvaluator:
    def test_default_metrics(self):
        evaluator = MultiGritsEvaluator()
        assert evaluator.metrics == ["con", "loc", "top"]

    def test_custom_metrics(self):
        evaluator = MultiGritsEvaluator(metrics=["con", "top"])
        assert evaluator.metrics == ["con", "top"]

    def test_eval_sample_perfect_score(self):
        evaluator = MultiGritsEvaluator(metrics=["con", "top"])
        true_grid_con = np.array([["A", "B"], ["C", "D"]], dtype=object)
        true_grid_top = [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]]
        true_grids = {"con": [true_grid_con], "top": [true_grid_top]}
        evaluator.eval_sample(true_grids, true_grids)
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0
            assert results[f"grits_{metric}_precision"] == 1.0
            assert results[f"grits_{metric}_recall"] == 1.0

    def test_compute_avg_grits(self):
        evaluator = MultiGritsEvaluator(metrics=["con"])
        grid = np.array([["A", "B"]], dtype=object)
        evaluator.eval_sample({"con": [grid]}, {"con": [grid]})
        results = evaluator.compute_avg_grits()
        assert results["avg_grits_con"] == 1.0
        assert results["avg_grits_con_precision"] == 1.0
        assert results["avg_grits_con_recall"] == 1.0

    def test_eval_html_perfect_score(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        evaluator = MultiGritsEvaluator()
        evaluator.eval_html(html, html)
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0
            assert results[f"grits_{metric}_precision"] == 1.0
            assert results[f"grits_{metric}_recall"] == 1.0

    def test_eval_html_different_tables(self):
        true_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        pred_html = "<table><tr><td>A</td><td>X</td></tr></table>"
        evaluator = MultiGritsEvaluator()
        evaluator.eval_html(true_html, pred_html)
        results = evaluator.compute_grits()
        # "top" should be perfect (same structure), "con" should not (different text)
        assert results["grits_top"] == 1.0
        assert results["grits_con"] < 1.0

    def test_eval_html_invalid_html_raises(self):
        evaluator = MultiGritsEvaluator()
        with pytest.raises(ValueError, match="Failed to parse true_html"):
            evaluator.eval_html("not valid html<><<", "<table><tr><td>A</td></tr></table>")
        with pytest.raises(ValueError, match="Failed to parse pred_html"):
            evaluator.eval_html("<table><tr><td>A</td></tr></table>", "not valid html<><<")

    def test_eval_html_list_perfect_score(self):
        html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
        html_b = "<table><tr><td>X</td><td>Y</td></tr></table>"
        evaluator = MultiGritsEvaluator()
        evaluator.eval_html([html_a, html_b], [html_a, html_b])
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_html_list_hungarian_matching(self):
        html_a = "<table><tr><td>A</td><td>B</td></tr></table>"
        html_b = "<table><tr><td>X</td><td>Y</td></tr></table>"
        evaluator = MultiGritsEvaluator()
        # Swapped order — Hungarian matching should pair them optimally
        evaluator.eval_html([html_a, html_b], [html_b, html_a])
        results = evaluator.compute_grits()
        for metric in ("con", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_html_list_invalid_raises(self):
        evaluator = MultiGritsEvaluator()
        with pytest.raises(ValueError, match="Failed to parse true_html"):
            evaluator.eval_html(
                ["<table><tr><td>A</td></tr></table>", "not valid<<>"],
                ["<table><tr><td>A</td></tr></table>"],
            )
        with pytest.raises(ValueError, match="Failed to parse pred_html"):
            evaluator.eval_html(
                ["<table><tr><td>A</td></tr></table>"],
                ["not valid<<>"],
            )

    def test_eval_html_missing_metric_raises(self):
        evaluator = MultiGritsEvaluator(metrics=["con"])
        with pytest.raises(ValueError, match="eval_html requires both"):
            evaluator.eval_html(
                "<table><tr><td>A</td></tr></table>", "<table><tr><td>A</td></tr></table>"
            )

    def test_omits_metrics_without_samples(self):
        evaluator = MultiGritsEvaluator()
        html = "<table><tr><td>A</td></tr></table>"
        evaluator.eval_html(html, html)
        results = evaluator.compute_grits()
        assert "grits_con" in results
        assert "grits_top" in results
        assert "grits_loc" not in results
        avg_results = evaluator.compute_avg_grits()
        assert "avg_grits_con" in avg_results
        assert "avg_grits_top" in avg_results
        assert "avg_grits_loc" not in avg_results

    def test_eval_table_cell_dicts_perfect(self):
        evaluator = MultiGritsEvaluator()
        cell_dicts = _make_simple_cell_dicts()
        evaluator.eval_table_cell_dicts([cell_dicts], [cell_dicts])
        results = evaluator.compute_grits()
        for metric in ("con", "loc", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_table_cell_dicts_different_text(self):
        evaluator = MultiGritsEvaluator()
        true_cell_dicts = _make_simple_cell_dicts()
        pred_cell_dicts = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "X", "bbox": [0, 0, 10, 10]},
            {"row_nums": [0], "column_nums": [1], "cell_text": "Y", "bbox": [10, 0, 20, 10]},
            {"row_nums": [1], "column_nums": [0], "cell_text": "Z", "bbox": [0, 10, 10, 20]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "W", "bbox": [10, 10, 20, 20]},
        ]
        evaluator.eval_table_cell_dicts([true_cell_dicts], [pred_cell_dicts])
        results = evaluator.compute_grits()
        assert results["grits_con"] < 1.0
        assert results["grits_loc"] == 1.0
        assert results["grits_top"] == 1.0

    def test_eval_table_cell_dicts_list(self):
        evaluator = MultiGritsEvaluator()
        cell_dicts_a = _make_simple_cell_dicts()
        cell_dicts_b = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "X", "bbox": [0, 0, 5, 5]},
        ]
        evaluator.eval_table_cell_dicts([cell_dicts_a, cell_dicts_b], [cell_dicts_a, cell_dicts_b])
        results = evaluator.compute_grits()
        for metric in ("con", "loc", "top"):
            assert results[f"grits_{metric}"] == 1.0

    def test_eval_table_cell_dicts_different_bbox(self):
        evaluator = MultiGritsEvaluator()
        true_cell_dicts = _make_simple_cell_dicts()
        # Same text and structure, but shifted bboxes
        pred_cell_dicts = [
            {"row_nums": [0], "column_nums": [0], "cell_text": "A", "bbox": [1, 1, 10, 10]},
            {"row_nums": [0], "column_nums": [1], "cell_text": "B", "bbox": [11, 1, 20, 10]},
            {"row_nums": [1], "column_nums": [0], "cell_text": "C", "bbox": [1, 11, 10, 20]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [11, 11, 20, 20]},
        ]
        evaluator.eval_table_cell_dicts([true_cell_dicts], [pred_cell_dicts])
        results = evaluator.compute_grits()
        assert results["grits_con"] == 1.0
        assert results["grits_loc"] == pytest.approx(0.81, abs=1e-4)
        assert results["grits_top"] == 1.0

    def test_eval_table_cell_dicts_different_structure(self):
        evaluator = MultiGritsEvaluator()
        true_cell_dicts = _make_simple_cell_dicts()
        # Same text and bboxes, but one cell spans two columns
        pred_cell_dicts = [
            {"row_nums": [0], "column_nums": [0, 1], "cell_text": "A", "bbox": [0, 0, 10, 10]},
            {"row_nums": [1], "column_nums": [0], "cell_text": "C", "bbox": [0, 10, 10, 20]},
            {"row_nums": [1], "column_nums": [1], "cell_text": "D", "bbox": [10, 10, 20, 20]},
        ]
        evaluator.eval_table_cell_dicts([true_cell_dicts], [pred_cell_dicts])
        results = evaluator.compute_grits()
        assert results["grits_top"] == 0.75

    def test_eval_table_cell_dicts_missing_metric_raises(self):
        evaluator = MultiGritsEvaluator(metrics=["con", "top"])
        cell_dicts = _make_simple_cell_dicts()
        with pytest.raises(ValueError, match="eval_table_cell_dicts requires"):
            evaluator.eval_table_cell_dicts([cell_dicts], [cell_dicts])

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            MultiGritsEvaluator(metrics=["invalid"])
