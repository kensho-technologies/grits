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

```python
from grits import grits_con, cells_to_grid

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

true_grid = cells_to_grid(true_cells, key="cell_text")
pred_grid = cells_to_grid(pred_cells, key="cell_text")

true_grid_scores, pred_grid_scores, true_positive_score, true_positive_score_upper = grits_con(true_grid, pred_grid)
```

### Evaluating across multiple tables

Use `GritsEvaluator` to accumulate results across a dataset and compute aggregate metrics:

```python
from grits import GritsEvaluator, cells_to_grid

evaluator = GritsEvaluator(metric="con")

for true_cells, pred_cells in dataset:
    true_grid = cells_to_grid(true_cells, key="cell_text")
    pred_grid = cells_to_grid(pred_cells, key="cell_text")
    evaluator.add(true_grid, pred_grid)

f1, precision, recall = evaluator.compute_f1()       # micro-averaged
avg_f1, avg_prec, avg_rec = evaluator.compute_avg_f1()  # macro-averaged
```

### Parsing HTML tables

```python
from grits import html_to_cells, cells_to_grid, cells_to_relspan_grid

cells = html_to_cells("<table><tr><td>A</td><td>B</td></tr></table>")
text_grid = cells_to_grid(cells, key="cell_text")
topology_grid = cells_to_relspan_grid(cells)
```

## Metrics

| Metric | Function | Measures |
|--------|----------|----------|
| GriTS_Con | `grits_con` | Cell text content similarity (using LCS) |
| GriTS_Top | `grits_top` | Cell topology / spanning structure (using IoU of relative spans) |
| GriTS_Loc | `grits_loc` | Cell spatial location similarity (using IoU of bounding boxes) |

For evaluating multiple tables on a page with optimal matching, use `hungarian_grits_con` and `hungarian_grits_top`.

## References

- B. Smock, R. Pesala, R. Abraham. [GriTS: Grid Table Similarity Metric for Table Structure Recognition](https://arxiv.org/abs/2203.12555). ICDAR 2023.
- B. Smock, V. Faucon-Morin, M. Sokolov, L. Liang, T. Khanam, A. Ramesh, M. Courtland. [PubTables-v2: A New Large-Scale Dataset for Full-Page and Multi-Page Table Extraction](https://arxiv.org/abs/2512.10888). 2025.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright 2025-present Kensho Technologies, LLC.
