# GriTS: Grid Table Similarity

GriTS is a Python package for evaluating table structure recognition (TSR) and table extraction (TE) using the Grid Table Similarity (GriTS) metric.

<img src="https://github.com/user-attachments/assets/2c42cbf4-9116-4370-a281-16ebaefbef92" width="80%">

<img src="https://github.com/user-attachments/assets/e0a9ce10-e8aa-4d29-b5a1-ff5c9141199c" height="100">

## About

The original GriTS metric was proposed in [GriTS: Grid Table Similarity Metric for Table Structure Recognition](https://arxiv.org/abs/2203.12555) for measuring the similarity between one predicted table and one ground truth table (the traditional TSR task). It treats each table as a matrix (grid) and computes a similarity between matrices (grids).

Subsequent work [PubTables-v2](https://arxiv.org/abs/2512.10888) generalized GriTS for page and document-level table extraction (TE). In this version, GriTS evaluates a list of predicted tables against a list of ground truth tables, assuming no correspondence is given between the two. GriTS determines the one-to-one correspondence that maximizes their aggregate similarity using the Hungarian algorithm.

In the case of one predicted table and one ground truth table (traditional TSR task), GriTS for TE is equivalent to GriTS for TSR.

But there are now two different ways to *aggregate* the score for an entire ground truth dataset.
1. Original way (macro F1 score): compute GriTS (which is a pseudo-F1 score) for each individual sample, then *average* the GriTS score over all samples.
2. New way (micro F1 score): compute the *true positive score* for each individual sample, then compute GriTS as the pseudo-F1 score for the total true positive score for the entire dataset.

We recommend aggregating GriTS the new way, which is the default choice in this package. The old way is supported for reproducing prior TSR work.

## Installation

```bash
pip install grits-metric
```

Requires Python >= 3.10.

## Quick start

### Computing GriTS-Con for an individual sample

```python
from grits import grits_con, compute_fscore, cells_to_grid_con

# Define ground-truth and predicted tables as lists of cell dicts
true_cells = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "95"},
]
pred_cells = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "90"},
]

true_grid_con = cells_to_grid_con(true_cells)
pred_grid_con = cells_to_grid_con(pred_cells)

result = grits_con(true_grid_con, pred_grid_con)  # returns a GritsResult

num_true_cells = len(true_grid_con) * len(true_grid_con[0])
num_pred_cells = len(pred_grid_con) * len(pred_grid_con[0])

grits_con_score, grits_con_precision, grits_con_recall = compute_fscore(result.true_positive_score, num_true_cells, num_pred_cells)

print(f"GriTS_Con: {grits_con_score:.4f}") # GriTS_Con: 0.8750
```

### Evaluating and aggregating across a collection of samples

Use `GritsEvaluator` to accumulate results across a dataset and compute aggregate metrics:

```python
from grits import GritsEvaluator, cells_to_grid_con

evaluator = GritsEvaluator(metric="con")

# Sample 1
true_cells_1 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "95"},
]
pred_cells_1 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "90"},
]

true_grid_con_1 = cells_to_grid_con(true_cells_1)
pred_grid_con_1 = cells_to_grid_con(pred_cells_1)
evaluator.eval_sample([true_grid_con_1], [pred_grid_con_1])

# Sample 2
true_cells_2 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Bob"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "85"},
]
pred_cells_2 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Bob"}
]

true_grid_con_2 = cells_to_grid_con(true_cells_2)
pred_grid_con_2 = cells_to_grid_con(pred_cells_2)
evaluator.eval_sample([true_grid_con_2], [pred_grid_con_2])

results = evaluator.compute_grits()           # micro (dict with grits_con, grits_con_precision, etc.)
avg_results = evaluator.compute_avg_grits()   # macro-averaged (dict with avg_grits_con, etc.)

print(f"GriTS_Con: {results['grits_con']:.4f}") # GriTS_Con: 0.6786
print(f"Avg GriTS_Con: {avg_results['avg_grits_con']:.4f}") # Avg GriTS_Con: 0.6458
```

### Evaluating multiple metrics at once

Use `MultiGritsEvaluator` to track multiple metrics simultaneously:

```python
from grits import MultiGritsEvaluator

evaluator = MultiGritsEvaluator()  # tracks "con", "loc", and "top" by default

true_html = "<table><tr><td>Name</td><td>Score</td></tr><tr><td>Alice</td><td>95</td></tr></table>"
pred_html = "<table><tr><td>Name</td><td>Score</td></tr><tr><td>Alice</td><td>90</td></tr></table>"

# From HTML (evaluates con and top):
evaluator.eval_html(true_html, pred_html)

results = evaluator.compute_grits()

print(f"GriTS_Top: {results['grits_top']:.4f}") # GriTS_Top: 1.0000
print(f"GriTS_Con: {results['grits_con']:.4f}") # GriTS_Con: 0.8750
```

### Converting table formats, from HTML tables to grids

```python
from grits import html_to_cell_dicts, cells_to_grid_con, cells_to_grid_top

cell_dicts = html_to_cell_dicts("<table><tr><td>A</td><td>B</td></tr></table>")
grid_con = cells_to_grid_con(cell_dicts)
grid_top = cells_to_grid_top(cell_dicts)

print(grid_con) # [['A', 'B']]
print(grid_top) # [[[0, 0, 1, 1], [0, 0, 1, 1]]]
```

## Metrics

| Metric | Function | Measures |
|--------|----------|----------|
| GriTS_Con | `grits_con` | Cell text content similarity (using LCS) |
| GriTS_Top | `grits_top` | Cell topology / spanning structure (using IoU of relative spans) |
| GriTS_Loc | `grits_loc` | Cell spatial location similarity (using IoU of bounding boxes) |

These functions return a `GritsResult` dataclass with named fields such as `true_positive_score`, `true_grid_scores`, and `is_exact_match`.

For evaluating multiple tables on a page with optimal matching, use `hungarian_grits_con`, `hungarian_grits_top`, and `hungarian_grits_loc`. These return a `HungarianGritsResult` dataclass with named fields such as `true_positive_score`, `matched_true_indices`, and `num_exact_grid_matches`.

## References

- B. Smock, R. Pesala, R. Abraham. [GriTS: Grid Table Similarity Metric for Table Structure Recognition](https://arxiv.org/abs/2203.12555). ICDAR 2023.
- B. Smock, V. Faucon-Morin, M. Sokolov, L. Liang, T. Khanam, A. Ramesh, M. Courtland. [PubTables-v2: A New Large-Scale Dataset for Full-Page and Multi-Page Table Extraction](https://arxiv.org/abs/2512.10888). 2025.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright 2025-present Kensho Technologies, LLC.
