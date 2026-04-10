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

Here we illustrate a basic example converting two tables to their grid representations, then calculating GriTS-Con.

```python
from grits import grits_con, cell_dicts_to_grid_con

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

true_grid_con = cell_dicts_to_grid_con(true_cells)
pred_grid_con = cell_dicts_to_grid_con(pred_cells)

grits_con_score, grits_con_precision, grits_con_recall = grits_con(true_grid_con, pred_grid_con)

print(f"GriTS_Con: {grits_con_score:.4f}") # GriTS_Con: 0.8750
```

The GriTS metric accepts table input as a grid (matrix).
There are three types of grids (Grid-Top, Grid-Con, Grid-Loc) containing different features of the table cells to be evaluated.
The library can convert a table to one or more grid types either from an HTML string representation or a list of cell dicts.

The above example illustrates converting each table to a grid (con) from the cell dicts representation.
Then GriTS-Con is computed between the two grids.

### Aggregating across a collection of samples

Use `GritsEvaluator` to accumulate results across a dataset and compute aggregate metrics.
In this example, we track a single metric type, GriTS-Con.

```python
from grits import GritsEvaluator, cell_dicts_to_grid_con

evaluator = GritsEvaluator(metrics=["con"])

true_samples = []
pred_samples = []

# Create two data samples
# Sample 1
true_cells_1 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "95"},
]
true_grid_con_1 = cell_dicts_to_grid_con(true_cells_1)
true_samples.append(true_grid_con_1)

pred_cells_1 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Alice"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "90"},
]
pred_grid_con_1 = cell_dicts_to_grid_con(pred_cells_1)
pred_samples.append(pred_grid_con_1)

# Sample 2
true_cells_2 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Score"},
    {"row_nums": [1], "column_nums": [0], "cell_text": "Bob"},
    {"row_nums": [1], "column_nums": [1], "cell_text": "85"},
]
true_grid_con_2 = cell_dicts_to_grid_con(true_cells_2)
true_samples.append(true_grid_con_2)

pred_cells_2 = [
    {"row_nums": [0], "column_nums": [0], "cell_text": "Name"},
    {"row_nums": [0], "column_nums": [1], "cell_text": "Bob"}
]
pred_grid_con_2 = cell_dicts_to_grid_con(pred_cells_2)
pred_samples.append(pred_grid_con_2)

# Iterate over samples and evaluate each
for true_grid_con, pred_grid_con in zip(true_samples, pred_samples):
    evaluator.eval_sample([true_grid_con], [pred_grid_con])

results = evaluator.compute_grits()

print(f"GriTS_Con: {results['grits_con']:.4f}") # GriTS_Con: 0.6786
```

### Evaluating multiple metrics at once

Use `GritsEvaluator` to track multiple metrics simultaneously:

```python
from grits import GritsEvaluator

evaluator = GritsEvaluator(metrics=["top", "con"])

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
from grits import html_to_cell_dicts, cell_dicts_to_grid_con, cell_dicts_to_grid_top

cell_dicts = html_to_cell_dicts("<table><tr><td>A</td><td>B</td></tr></table>")
grid_con = cell_dicts_to_grid_con(cell_dicts)
grid_top = cell_dicts_to_grid_top(cell_dicts)

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

For evaluating multiple tables on a page with optimal matching, use `hungarian_grits_con_matching`, `hungarian_grits_top_matching`, and `hungarian_grits_loc_matching`. These return a `HungarianGritsResult` dataclass with named fields such as `true_positive_score`, `matched_true_indices`, and `num_exact_grid_matches`.

## References

- B. Smock, R. Pesala, R. Abraham. [GriTS: Grid Table Similarity Metric for Table Structure Recognition](https://arxiv.org/abs/2203.12555). ICDAR 2023.
- B. Smock, V. Faucon-Morin, M. Sokolov, L. Liang, T. Khanam, A. Ramesh, M. Courtland. [PubTables-v2: A New Large-Scale Dataset for Full-Page and Multi-Page Table Extraction](https://arxiv.org/abs/2512.10888). 2025.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright 2025-present Kensho Technologies, LLC.
