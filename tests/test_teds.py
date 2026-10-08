# Copyright 2025-present Kensho Technologies, LLC.
import pytest

from grits._teds import (
    TedsEvaluator,
    _TedsMetricEvaluator,
    hungarian_teds,
    hungarian_teds_matching,
    hungarian_teds_struct,
    hungarian_teds_struct_matching,
    teds,
    teds_struct,
)


T2X2 = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"

# (name, true_html, pred_html, expected TEDS, expected TEDS-Struct).
# Expected values were computed with the reference implementation in IBM's PubTabNet
# repository (src/metric.py), with each table wrapped in <html> and exceptions scored as 0.
REFERENCE_PAIRS = [
    ("identical", T2X2, T2X2, 1.0, 1.0),
    (
        "cell_text_replaced",
        T2X2,
        "<table><tr><td>A</td><td>X</td></tr><tr><td>C</td><td>D</td></tr></table>",
        0.8333333333333334,
        1.0,
    ),
    (
        "cell_text_partial_edit",
        "<table><tr><td>Revenue</td><td>2020</td></tr><tr><td>Costs</td><td>2021</td></tr></table>",
        "<table><tr><td>Revenues</td><td>2020</td></tr><tr><td>Cost</td><td>2012</td></tr></table>",
        0.8625,
        1.0,
    ),
    (
        "empty_vs_filled_cell",
        T2X2,
        "<table><tr><td>A</td><td></td></tr><tr><td>C</td><td>D</td></tr></table>",
        0.8333333333333334,
        1.0,
    ),
    (
        "colspan_change",
        '<table><tr><td colspan="2">AB</td></tr><tr><td>C</td><td>D</td></tr></table>',
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        0.6666666666666667,
        0.6666666666666667,
    ),
    (
        "rowspan_change",
        '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>D</td></tr></table>',
        "<table><tr><td>A</td><td>B</td></tr><tr><td>D</td></tr></table>",
        0.8,
        0.8,
    ),
    (
        "explicit_default_spans",
        T2X2,
        '<table><tr><td colspan="1">A</td><td rowspan="1">B</td></tr><tr><td>C</td><td>D</td></tr></table>',
        1.0,
        1.0,
    ),
    ("missing_row", T2X2, "<table><tr><td>A</td><td>B</td></tr></table>", 0.5, 0.5),
    (
        "extra_column",
        T2X2,
        "<table><tr><td>A</td><td>B</td><td>E</td></tr><tr><td>C</td><td>D</td><td>F</td></tr></table>",
        0.75,
        0.75,
    ),
    (
        "inline_bold",
        T2X2,
        "<table><tr><td><b>A</b></td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        0.9047619047619048,
        1.0,
    ),
    (
        "inline_sup_and_br",
        "<table><tr><td>x2</td><td>a b</td></tr></table>",
        "<table><tr><td>x<sup>2</sup></td><td>a<br>b</td></tr></table>",
        0.8,
        1.0,
    ),
    (
        "inline_with_tail_text",
        "<table><tr><td>abc</td></tr></table>",
        "<table><tr><td>a<i>b</i>c</td></tr></table>",
        0.8666666666666667,
        1.0,
    ),
    (
        "th_text_change",
        "<table><tr><th>Header</th></tr><tr><td>A</td></tr></table>",
        "<table><tr><th>Different</th></tr><tr><td>A</td></tr></table>",
        1.0,
        1.0,
    ),
    (
        "th_colspan_change",
        '<table><tr><th colspan="2">H</th></tr><tr><td>A</td><td>B</td></tr></table>',
        '<table><tr><th colspan="3">H</th></tr><tr><td>A</td><td>B</td></tr></table>',
        1.0,
        1.0,
    ),
    (
        "th_with_inline_tag",
        "<table><tr><th>H</th></tr><tr><td>A</td></tr></table>",
        "<table><tr><th><b>H</b></th></tr><tr><td>A</td></tr></table>",
        0.8,
        0.8,
    ),
    (
        "th_vs_td",
        "<table><tr><th>H</th></tr><tr><td>A</td></tr></table>",
        "<table><tr><td>H</td></tr><tr><td>A</td></tr></table>",
        0.75,
        0.75,
    ),
    (
        "thead_tbody_vs_none",
        "<table><thead><tr><td>H</td></tr></thead><tbody><tr><td>A</td></tr></tbody></table>",
        "<table><tr><td>H</td></tr><tr><td>A</td></tr></table>",
        0.6666666666666667,
        0.6666666666666667,
    ),
    (
        "whitespace_formatting",
        T2X2,
        "<table>\n  <tr>\n    <td>A</td>\n    <td>B</td>\n  </tr>\n  <tr>\n    <td>C</td>\n    <td>D</td>\n  </tr>\n</table>",
        1.0,
        1.0,
    ),
    (
        "entities_and_comments",
        "<table><tr><td>A&amp;B</td><td>x y</td></tr></table>",
        "<table><tr><td>A&amp;B<!-- note --></td><td>x&nbsp;y</td></tr></table>",
        0.8888888888888888,
        1.0,
    ),
    (
        "unicode_content",
        "<table><tr><td>Größe</td><td>€5</td></tr></table>",
        "<table><tr><td>Grosse</td><td>€5</td></tr></table>",
        0.8333333333333334,
        1.0,
    ),
    (
        "completely_different",
        T2X2,
        "<table><tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr><tr><td>4</td></tr><tr><td>5</td></tr></table>",
        0.0,
        0.19999999999999996,
    ),
    ("empty_pred_string", T2X2, "", 0.0, 0.0),
    ("pred_without_table", T2X2, "<p>no table here</p>", 0.0, 0.0),
    ("both_tables_without_children", "<table></table>", "<table></table>", 0.0, 0.0),
    ("one_table_without_children", T2X2, "<table></table>", 0.0, 0.0),
    (
        "invalid_colspan",
        T2X2,
        '<table><tr><td colspan="abc">A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
        0.0,
        0.0,
    ),
]


def _wrap(table_html):
    return f"<html>{table_html}</html>" if table_html else table_html


class TestTedsReference:
    @pytest.mark.parametrize(
        "true_html,pred_html,expected",
        [(t, p, e) for _, t, p, e, _ in REFERENCE_PAIRS],
        ids=[name for name, *_ in REFERENCE_PAIRS],
    )
    def test_teds(self, true_html, pred_html, expected):
        assert teds(true_html, pred_html) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(
        "true_html,pred_html,expected",
        [(t, p, e) for _, t, p, _, e in REFERENCE_PAIRS],
        ids=[name for name, *_ in REFERENCE_PAIRS],
    )
    def test_teds_struct(self, true_html, pred_html, expected):
        assert teds_struct(true_html, pred_html) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize(
        "true_html,pred_html",
        [(t, p) for _, t, p, *_ in REFERENCE_PAIRS],
        ids=[name for name, *_ in REFERENCE_PAIRS],
    )
    def test_wrapped_input_matches_bare_input(self, true_html, pred_html):
        assert teds(_wrap(true_html), _wrap(pred_html)) == teds(true_html, pred_html)
        assert teds_struct(_wrap(true_html), _wrap(pred_html)) == teds_struct(true_html, pred_html)


class TestTedsQuirks:
    """Pin the reference implementation's quirks so they are not changed by accident."""

    def test_inline_tags_increase_struct_score(self):
        # Inline tags count toward the normalizer but are not part of the tree, so they raise
        # TEDS-Struct for an otherwise identical structural error.
        pred = "<table><tr><td>A</td><td>B</td></tr></table>"
        bold_true = "<table><tr><td><b>A</b></td><td><b>B</b></td></tr><tr><td>C</td><td>D</td></tr></table>"
        assert teds_struct(T2X2, pred) == pytest.approx(0.5)
        assert teds_struct(bold_true, pred) == pytest.approx(1 - 3 / 8)

    def test_th_text_is_ignored(self):
        true = "<table><tr><th>Header</th></tr><tr><td>A</td></tr></table>"
        pred = "<table><tr><th>Other</th></tr><tr><td>A</td></tr></table>"
        assert teds(true, pred) == 1.0

    def test_th_spans_are_ignored(self):
        true = '<table><tr><th colspan="2">H</th></tr><tr><td>A</td><td>B</td></tr></table>'
        pred = '<table><tr><th rowspan="3">H</th></tr><tr><td>A</td><td>B</td></tr></table>'
        assert teds_struct(true, pred) == 1.0

    def test_only_first_table_is_scored(self):
        other = "<table><tr><td>Z</td></tr></table>"
        assert teds(T2X2 + other, T2X2) == 1.0

    def test_inline_tags_are_single_content_tokens(self):
        # "<b>" and "</b>" are one token each, so the cell content differs by 2 of 3 tokens
        true = "<table><tr><td>A</td></tr></table>"
        pred = "<table><tr><td><b>A</b></td></tr></table>"
        assert teds(true, pred) == pytest.approx(1 - (2 / 3) / 3)

    def test_argument_order_does_not_matter(self):
        _, true_html, pred_html, *_ = REFERENCE_PAIRS[2]
        assert teds(true_html, pred_html) == pytest.approx(teds(pred_html, true_html))


T_A = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Ann</td><td>34</td></tr></table>"
T_A_PRED = "<table><tr><td>Name</td><td>Age</td></tr><tr><td>Anne</td><td>34</td></tr></table>"
T_B = '<table><tr><td colspan="2">Totals</td></tr><tr><td>Q1</td><td>Q2</td></tr><tr><td>10</td><td>20</td></tr></table>'
T_B_PRED = "<table><tr><td>Totals</td><td></td></tr><tr><td>Q1</td><td>Q2</td></tr><tr><td>10</td><td>20</td></tr></table>"
T_C = "<table><tr><td>x</td></tr></table>"
# T_A's structure with wrong content, and T_A's content with an extra empty column
T_A_WRONG_TEXT = (
    "<table><tr><td>qqqqqqq</td><td>zzzz</td></tr><tr><td>wwww</td><td>vv</td></tr></table>"
)
T_A_EXTRA_COL = "<table><tr><td>Name</td><td>Age</td><td></td></tr><tr><td>Ann</td><td>34</td><td></td></tr></table>"

# name -> (true tables, pred tables). Totals were computed with the reference implementation and
# the Hungarian matching in the reference evaluation script.
REFERENCE_SAMPLES = {
    "swapped_order": ([T_A, T_B], [T_B_PRED, T_A_PRED]),
    "extra_pred": ([T_A], [T_A, T_C]),
    "missing_preds": ([T_A, T_B], []),
    "no_tables": ([], []),
    "different_matchings": ([T_A], [T_A_WRONG_TEXT, T_A_EXTRA_COL]),
}


class TestHungarianTeds:
    def test_swapped_order(self):
        true_htmls, pred_htmls = REFERENCE_SAMPLES["swapped_order"]
        result = hungarian_teds_matching(true_htmls, pred_htmls)
        assert result.true_positive_score == pytest.approx(1.7361111111111112)
        assert result.matched_true_indices == [0, 1]
        assert result.matched_pred_indices == [1, 0]
        assert result.num_exact_table_matches == 0
        result = hungarian_teds_struct_matching(true_htmls, pred_htmls)
        assert result.true_positive_score == pytest.approx(1.7777777777777777)

    def test_extra_pred_reduces_precision(self):
        true_htmls, pred_htmls = REFERENCE_SAMPLES["extra_pred"]
        result = hungarian_teds_matching(true_htmls, pred_htmls)
        assert result.true_positive_score == 1.0
        assert result.matched_pred_indices == [0]
        assert result.num_exact_table_matches == 1
        f1, precision, recall = hungarian_teds(true_htmls, pred_htmls)
        assert (precision, recall) == (0.5, 1.0)
        assert f1 == pytest.approx(2 / 3)

    def test_missing_preds_reduce_recall(self):
        true_htmls, pred_htmls = REFERENCE_SAMPLES["missing_preds"]
        result = hungarian_teds_matching(true_htmls, pred_htmls)
        assert result.true_positive_score == 0.0
        assert result.matched_true_indices == []
        assert hungarian_teds(true_htmls, pred_htmls) == (0.0, 1.0, 0.0)

    def test_no_tables(self):
        assert hungarian_teds([], []) == (1.0, 1.0, 1.0)
        assert hungarian_teds_struct([], []) == (1.0, 1.0, 1.0)

    def test_teds_and_teds_struct_match_separately(self):
        true_htmls, pred_htmls = REFERENCE_SAMPLES["different_matchings"]
        content_result = hungarian_teds_matching(true_htmls, pred_htmls)
        struct_result = hungarian_teds_struct_matching(true_htmls, pred_htmls)
        assert content_result.matched_pred_indices == [1]
        assert content_result.true_positive_score == pytest.approx(0.75)
        assert struct_result.matched_pred_indices == [0]
        assert struct_result.true_positive_score == 1.0


class TestTedsMetricEvaluator:
    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            _TedsMetricEvaluator(metric="con")

    def test_accumulates_samples(self):
        evaluator = _TedsMetricEvaluator(metric="teds")
        evaluator.eval_htmls([T_A], [T_A, T_C])
        evaluator.eval_htmls([T_A, T_B], [])
        assert evaluator.true_positive_scores == [1.0, 0.0]
        assert evaluator.num_true_tables == [1, 2]
        assert evaluator.num_pred_tables == [2, 0]
        assert evaluator.num_exact_table_matches == [1, 0]

    def test_no_true_tables(self):
        evaluator = _TedsMetricEvaluator(metric="teds")
        evaluator.eval_htmls([], [T_A])
        results = evaluator.compute_teds()
        assert results["teds_precision"] == 0.0
        assert results["teds_recall"] == 1.0
        assert results["teds_table_exact_match_accuracy"] == 1.0
        results = evaluator.compute_mean_teds_per_sample()
        assert results["mean_teds_table_exact_match_accuracy_per_sample"] == 1.0

    def test_samples_with_no_true_tables_are_skipped_in_accuracy_average(self):
        evaluator = _TedsMetricEvaluator(metric="teds")
        evaluator.eval_htmls([T_A], [T_A_PRED])
        evaluator.eval_htmls([], [T_A])
        assert (
            evaluator.compute_mean_teds_per_sample()[
                "mean_teds_table_exact_match_accuracy_per_sample"
            ]
            == 0.0
        )
        assert evaluator.compute_teds()["teds_table_exact_match_accuracy"] == 0.0


class TestTedsEvaluator:
    def _evaluate_reference_samples(self, metrics=None):
        evaluator = TedsEvaluator(metrics=metrics)
        for true_htmls, pred_htmls in REFERENCE_SAMPLES.values():
            evaluator.eval_htmls(true_htmls, pred_htmls)
        return evaluator

    def test_default_metrics(self):
        assert TedsEvaluator().metrics == ["teds", "teds_struct"]

    def test_counts(self):
        evaluator = self._evaluate_reference_samples()
        assert evaluator.compute_counts() == {
            "num_samples": 5,
            "num_true_tables": 6,
            "num_pred_tables": 6,
        }

    def test_micro_scores_match_reference_script_totals(self):
        results = self._evaluate_reference_samples().compute_teds()
        assert results["teds"] == pytest.approx(0.5810185185185185)
        assert results["teds_precision"] == pytest.approx(0.5810185185185185)
        assert results["teds_recall"] == pytest.approx(0.5810185185185185)
        assert results["teds_struct"] == pytest.approx(0.6296296296296297)
        assert results["teds_struct_precision"] == pytest.approx(0.6296296296296297)
        assert results["teds_struct_recall"] == pytest.approx(0.6296296296296297)
        assert results["teds_table_exact_match_accuracy"] == pytest.approx(1 / 6)
        assert results["teds_struct_table_exact_match_accuracy"] == pytest.approx(3 / 6)

    def test_macro_scores(self):
        results = self._evaluate_reference_samples().compute_mean_teds_per_sample()
        # The no_tables sample scores 1/1/1 per compute_fscore conventions
        assert results["mean_teds_per_sample"] == pytest.approx(0.6069444444444445)
        assert results["mean_teds_precision_per_sample"] == pytest.approx(0.7486111111111111)
        assert results["mean_teds_recall_per_sample"] == pytest.approx(0.7236111111111111)
        assert "mean_teds_struct_per_sample" in results

    def test_single_metric(self):
        results = self._evaluate_reference_samples(metrics=["teds_struct"]).compute_teds()
        assert set(results) == {
            "teds_struct",
            "teds_struct_precision",
            "teds_struct_recall",
            "teds_struct_table_exact_match_accuracy",
        }

    def test_no_samples(self):
        evaluator = TedsEvaluator()
        assert evaluator.compute_teds() == {}
        assert evaluator.compute_mean_teds_per_sample() == {}
        assert evaluator.compute_counts() == {
            "num_samples": 0,
            "num_true_tables": 0,
            "num_pred_tables": 0,
        }

    def test_bare_string_raises(self):
        with pytest.raises(TypeError, match="lists of HTML strings"):
            TedsEvaluator().eval_htmls(T2X2, [T2X2])
