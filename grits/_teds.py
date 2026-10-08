# Copyright 2025-present Kensho Technologies, LLC.
#
# Portions of this file are derived from the TEDS implementation in IBM's PubTabNet repository
# (https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py),
# Copyright 2020 IBM, licensed under the Apache License, Version 2.0.
# A copy of the license is in LICENSES/Apache-2.0.txt.
# Modified by Kensho Technologies: reimplemented with type annotations and true-then-pred
# argument order, uses rapidfuzz for edit distance, accepts bare <table> input, and adds Hungarian
# matching and aggregation over lists of tables.
"""Tree-Edit-Distance-based Similarity (TEDS) metrics.

TEDS was proposed in "Image-based table recognition: data, model, and evaluation"
(Zhong et al., 2020, https://arxiv.org/abs/1911.10683). This module is a reimplementation
following the reference implementation in IBM's PubTabNet repository
(https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py), and reproduces its scores
exactly, including two of its quirks, which are kept so that scores remain comparable with
published results:

1. The normalizer counts all descendant elements of the table, including inline elements
   inside cells (e.g. <b>, <sup>, <br>), while the edit distance is computed on a tree that
   stops at each td. Inline formatting therefore increases the score, also for TEDS-Struct.
2. Only td elements are treated as cells. A th element is an ordinary node: its text,
   colspan, and rowspan are ignored, and its inline elements become tree nodes.

One deliberate deviation: a bare <table>...</table> string is parsed as a full HTML document,
whereas the reference implementation scores input without an <html><body> wrapper as 0.
This does not change the score of any input the reference implementation can score.
"""

import dataclasses
from typing import Callable

from apted import APTED, Config
import lxml.etree
import lxml.html
import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .grits import compute_fscore


class _TableTree:
    """A node in the tree representation of an HTML table used by APTED."""

    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(
        self,
        tag: str,
        colspan: int | None = None,
        rowspan: int | None = None,
        content: list[str] | None = None,
    ) -> None:
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children: list[_TableTree] = []


class _TedsConfig(Config):
    """APTED cost configuration matching the reference TEDS implementation."""

    def rename(self, node1: _TableTree, node2: _TableTree) -> float:
        """Compute the cost of renaming node1 to node2."""
        if (
            node1.tag != node2.tag
            or node1.colspan != node2.colspan
            or node1.rowspan != node2.rowspan
        ):
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            content1 = node1.content or []
            content2 = node2.content or []
            distance = Levenshtein.distance(content1, content2)
            return distance / max(len(content1), len(content2))
        return 0.0

    def children(self, node: _TableTree) -> list[_TableTree]:
        """Return the children of a node."""
        return node.children


def _tokenize(element: lxml.html.HtmlElement, tokens: list[str]) -> None:
    """Append the tokens of an element: its tags, plus each character of its text."""
    tokens.append(f"<{element.tag}>")
    if element.text is not None:
        tokens.extend(element.text)
    for child in element:
        _tokenize(child, tokens)
    if element.tag != "unk":
        tokens.append(f"</{element.tag}>")
    if element.tag != "td" and element.tail is not None:
        tokens.extend(element.tail)


def _build_tree(element: lxml.html.HtmlElement, structure_only: bool) -> _TableTree:
    """Convert an lxml element to a _TableTree, stopping at td elements.

    Raises:
        ValueError: If a td has a non-integer colspan or rowspan.
    """
    if element.tag == "td":
        content: list[str] = []
        if not structure_only:
            _tokenize(element, content)
            content = content[1:-1]
        return _TableTree(
            element.tag,
            int(element.attrib.get("colspan", "1")),
            int(element.attrib.get("rowspan", "1")),
            content,
        )
    node = _TableTree(element.tag)
    node.children = [_build_tree(child, structure_only) for child in element]
    return node


@dataclasses.dataclass(frozen=True)
class _PreparedTable:
    """A table converted to the inputs TEDS needs: its tree and its node count."""

    tree: _TableTree
    num_nodes: int


def _prepare_table(table_html: str, structure_only: bool) -> _PreparedTable | None:
    """Parse an HTML table for TEDS.

    Returns None if the input is empty, contains no table, or has an invalid span attribute,
    which the reference implementation scores as 0.
    """
    if not table_html:
        return None
    try:
        # A new parser per call, since lxml parsers should not be shared between threads
        parser = lxml.html.HTMLParser(remove_comments=True, encoding="utf-8")
        document = lxml.html.document_fromstring(table_html, parser=parser)
    except (lxml.etree.ParserError, ValueError):
        return None
    tables = document.xpath("body/table")
    if not tables:
        return None
    table = tables[0]
    try:
        tree = _build_tree(table, structure_only)
    except ValueError:
        return None
    return _PreparedTable(tree=tree, num_nodes=len(table.xpath(".//*")))


def _teds_prepared(true_table: _PreparedTable | None, pred_table: _PreparedTable | None) -> float:
    """Compute TEDS between two prepared tables."""
    if true_table is None or pred_table is None:
        return 0.0
    num_nodes = max(true_table.num_nodes, pred_table.num_nodes)
    if num_nodes == 0:
        return 0.0
    distance = APTED(pred_table.tree, true_table.tree, _TedsConfig()).compute_edit_distance()
    return 1.0 - float(distance) / num_nodes


def teds(true_html: str, pred_html: str) -> float:
    """Compute TEDS between a ground truth table and a predicted table in HTML format.

    Args:
        true_html: HTML string of the ground truth table
        pred_html: HTML string of the predicted table

    Returns:
        The TEDS score. 0 if either table is empty, contains no table, or cannot be parsed.
    """
    return _teds_prepared(
        _prepare_table(true_html, structure_only=False),
        _prepare_table(pred_html, structure_only=False),
    )


def teds_struct(true_html: str, pred_html: str) -> float:
    """Compute TEDS-Struct (TEDS ignoring cell content) between two tables in HTML format.

    Args:
        true_html: HTML string of the ground truth table
        pred_html: HTML string of the predicted table

    Returns:
        The TEDS-Struct score. 0 if either table is empty, contains no table, or cannot be parsed.
    """
    return _teds_prepared(
        _prepare_table(true_html, structure_only=True),
        _prepare_table(pred_html, structure_only=True),
    )


@dataclasses.dataclass(frozen=True)
class HungarianTedsMatchingResult:
    """Result of computing TEDS across multiple tables using Hungarian matching."""

    true_positive_score: float
    pair_scores: np.ndarray
    matched_true_indices: list[int]
    matched_pred_indices: list[int]
    num_exact_table_matches: int


def _hungarian_teds_matching(
    true_htmls: list[str], pred_htmls: list[str], structure_only: bool
) -> HungarianTedsMatchingResult:
    """Match true and predicted tables to maximize the total TEDS score.

    Args:
        true_htmls: List of HTML strings of ground truth tables
        pred_htmls: List of HTML strings of predicted tables
        structure_only: Whether to ignore cell content (TEDS-Struct)

    Returns:
        HungarianTedsMatchingResult with the matching results.
    """
    true_tables = [_prepare_table(html, structure_only) for html in true_htmls]
    pred_tables = [_prepare_table(html, structure_only) for html in pred_htmls]

    score_matrix = np.zeros((len(true_tables), len(pred_tables)))
    for true_idx, true_table in enumerate(true_tables):
        for pred_idx, pred_table in enumerate(pred_tables):
            score_matrix[true_idx, pred_idx] = _teds_prepared(true_table, pred_table)

    matched_true_indices, matched_pred_indices = linear_sum_assignment(score_matrix, maximize=True)
    pair_scores = score_matrix[matched_true_indices, matched_pred_indices]

    return HungarianTedsMatchingResult(
        true_positive_score=float(pair_scores.sum()),
        pair_scores=pair_scores,
        matched_true_indices=[int(i) for i in matched_true_indices],
        matched_pred_indices=[int(i) for i in matched_pred_indices],
        num_exact_table_matches=int((pair_scores == 1).sum()),
    )


def hungarian_teds_matching(
    true_htmls: list[str], pred_htmls: list[str]
) -> HungarianTedsMatchingResult:
    """Compute TEDS across multiple tables using Hungarian algorithm for optimal matching.

    Args:
        true_htmls: List of HTML strings of ground truth tables
        pred_htmls: List of HTML strings of predicted tables

    Returns:
        HungarianTedsMatchingResult with the matching results.
    """
    return _hungarian_teds_matching(true_htmls, pred_htmls, structure_only=False)


def hungarian_teds_struct_matching(
    true_htmls: list[str], pred_htmls: list[str]
) -> HungarianTedsMatchingResult:
    """Compute TEDS-Struct across multiple tables using Hungarian algorithm for optimal matching.

    Args:
        true_htmls: List of HTML strings of ground truth tables
        pred_htmls: List of HTML strings of predicted tables

    Returns:
        HungarianTedsMatchingResult with the matching results.
    """
    return _hungarian_teds_matching(true_htmls, pred_htmls, structure_only=True)


def hungarian_teds(true_htmls: list[str], pred_htmls: list[str]) -> tuple[float, float, float]:
    """Compute the TEDS F-score, precision, and recall across multiple tables.

    Each matched pair of tables contributes its TEDS score; unmatched tables contribute 0.
    Precision and recall divide the total by the number of predicted and true tables.

    Args:
        true_htmls: List of HTML strings of ground truth tables
        pred_htmls: List of HTML strings of predicted tables

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = hungarian_teds_matching(true_htmls, pred_htmls)
    return compute_fscore(result.true_positive_score, len(true_htmls), len(pred_htmls))


def hungarian_teds_struct(
    true_htmls: list[str], pred_htmls: list[str]
) -> tuple[float, float, float]:
    """Compute the TEDS-Struct F-score, precision, and recall across multiple tables.

    Each matched pair of tables contributes its TEDS-Struct score; unmatched tables contribute 0.
    Precision and recall divide the total by the number of predicted and true tables.

    Args:
        true_htmls: List of HTML strings of ground truth tables
        pred_htmls: List of HTML strings of predicted tables

    Returns:
        A tuple containing (F1-score, Precision, Recall).
    """
    result = hungarian_teds_struct_matching(true_htmls, pred_htmls)
    return compute_fscore(result.true_positive_score, len(true_htmls), len(pred_htmls))


class _TedsMetricEvaluator:
    """Accumulates results from TEDS evaluation calls for a single metric and computes aggregate scores."""

    _METRIC_FUNCTIONS: dict[str, Callable[[list[str], list[str]], HungarianTedsMatchingResult]] = {
        "teds": hungarian_teds_matching,
        "teds_struct": hungarian_teds_struct_matching,
    }

    def __init__(self, metric: str = "teds") -> None:
        """Initialize the evaluator with the specified metric."""
        if metric not in self._METRIC_FUNCTIONS:
            raise ValueError(f"metric must be one of {set(self._METRIC_FUNCTIONS)}, got {metric!r}")
        self._metric = metric
        self._teds_function = self._METRIC_FUNCTIONS[metric]
        self.true_positive_scores: list[float] = []
        self.num_true_tables: list[int] = []
        self.num_pred_tables: list[int] = []
        self.num_exact_table_matches: list[int] = []

    def eval_htmls(self, true_htmls: list[str], pred_htmls: list[str]) -> None:
        """Evaluate a sample and store the results."""
        result = self._teds_function(true_htmls, pred_htmls)
        self.true_positive_scores.append(result.true_positive_score)
        self.num_true_tables.append(len(true_htmls))
        self.num_pred_tables.append(len(pred_htmls))
        self.num_exact_table_matches.append(result.num_exact_table_matches)

    def compute_counts(self) -> dict[str, int]:
        """Compute the counts of samples and tables.

        Returns:
            Dict with keys num_samples, num_true_tables, and num_pred_tables.
        """
        return {
            "num_samples": len(self.true_positive_scores),
            "num_true_tables": sum(self.num_true_tables),
            "num_pred_tables": sum(self.num_pred_tables),
        }

    def compute_teds(self) -> dict[str, float]:
        """Compute the (micro) TEDS score, which is a pseudo F1 score over all tables.

        Returns:
            Dict with keys METRIC, METRIC_precision, METRIC_recall, and
            METRIC_table_exact_match_accuracy. The exact match accuracy is the fraction of ground
            truth tables that are matched exactly, like a recall. If there are no ground truth
            tables, it is 1.0, following the same convention as recall in compute_fscore.
        """
        total_num_true = sum(self.num_true_tables)
        f1, precision, recall = compute_fscore(
            sum(self.true_positive_scores), total_num_true, sum(self.num_pred_tables)
        )
        # Like recall, exact match accuracy is 1.0 when there are no ground truth tables
        table_exact_match_accuracy = (
            sum(self.num_exact_table_matches) / total_num_true if total_num_true > 0 else 1.0
        )
        m = self._metric
        return {
            m: f1,
            f"{m}_precision": precision,
            f"{m}_recall": recall,
            f"{m}_table_exact_match_accuracy": table_exact_match_accuracy,
        }

    def compute_mean_teds_per_sample(self) -> dict[str, float]:
        """Compute the (macro) average TEDS score. Compute the TEDS score for each sample, then average over all samples.

        Returns:
            Dict with keys mean_METRIC_per_sample, mean_METRIC_precision_per_sample,
            mean_METRIC_recall_per_sample, and mean_METRIC_table_exact_match_accuracy_per_sample.
            The exact match accuracy is averaged over samples with ground truth tables; if no
            sample has ground truth tables, it is 1.0, as in compute_teds.
        """
        f1s = []
        precisions = []
        recalls = []
        table_exact_match_accuracies = []
        for true_positive_score, num_true, num_pred, num_exact in zip(
            self.true_positive_scores,
            self.num_true_tables,
            self.num_pred_tables,
            self.num_exact_table_matches,
        ):
            f1, precision, recall = compute_fscore(true_positive_score, num_true, num_pred)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            if num_true > 0:
                table_exact_match_accuracies.append(num_exact / num_true)
        m = self._metric
        # Like recall, exact match accuracy is 1.0 when there are no ground truth tables
        return {
            f"mean_{m}_per_sample": sum(f1s) / len(f1s),
            f"mean_{m}_precision_per_sample": sum(precisions) / len(precisions),
            f"mean_{m}_recall_per_sample": sum(recalls) / len(recalls),
            f"mean_{m}_table_exact_match_accuracy_per_sample": (
                sum(table_exact_match_accuracies) / len(table_exact_match_accuracies)
                if table_exact_match_accuracies
                else 1.0
            ),
        }


class TedsEvaluator:
    """Evaluates a set of TEDS metrics one-sample-at-a-time and computes aggregate metrics over all samples."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialize the evaluator with the specified metrics.

        Args:
            metrics: List of metric names to track from among ["teds", "teds_struct"]. Defaults to both.
        """
        if metrics is None:
            metrics = ["teds", "teds_struct"]
        self._evaluators = {metric: _TedsMetricEvaluator(metric=metric) for metric in metrics}

    @property
    def metrics(self) -> list[str]:
        """Return the list of metric names being tracked."""
        return list(self._evaluators)

    def eval_htmls(self, true_htmls: list[str], pred_htmls: list[str]) -> None:
        """Evaluate two table collections, with each table represented in HTML format.

        Args:
            true_htmls: List of HTML strings of ground truth tables.
            pred_htmls: List of HTML strings of predicted tables.

        Raises:
            TypeError: If a bare string is passed instead of a list of strings.
        """
        if isinstance(true_htmls, str) or isinstance(pred_htmls, str):  # type: ignore[unreachable]
            raise TypeError("eval_htmls expects lists of HTML strings, not a single string.")
        for evaluator in self._evaluators.values():
            evaluator.eval_htmls(true_htmls, pred_htmls)

    def compute_counts(self) -> dict[str, int]:
        """Compute the counts of samples and tables.

        Returns:
            Dict with keys num_samples, num_true_tables, and num_pred_tables.
        """
        for evaluator in self._evaluators.values():
            return evaluator.compute_counts()
        return {"num_samples": 0, "num_true_tables": 0, "num_pred_tables": 0}

    def compute_teds(self) -> dict[str, float]:
        """Compute micro TEDS scores for each metric.

        Returns:
            Dict with keys like teds, teds_precision, teds_recall for each metric
            with at least one evaluated sample.
        """
        results: dict[str, float] = {}
        for evaluator in self._evaluators.values():
            if evaluator.true_positive_scores:
                results.update(evaluator.compute_teds())
        return results

    def compute_mean_teds_per_sample(self) -> dict[str, float]:
        """Compute macro-averaged TEDS scores for each metric.

        Returns:
            Dict with keys like mean_teds_per_sample, mean_teds_precision_per_sample,
            mean_teds_recall_per_sample for each metric with at least one evaluated sample.
        """
        results: dict[str, float] = {}
        for evaluator in self._evaluators.values():
            if evaluator.true_positive_scores:
                results.update(evaluator.compute_mean_teds_per_sample())
        return results
