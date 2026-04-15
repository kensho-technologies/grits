# Copyright 2025-present Kensho Technologies, LLC.
from collections import defaultdict
import dataclasses
from typing import Any
import xml.etree.ElementTree as ET


@dataclasses.dataclass(frozen=True, slots=True)
class TableCell:
    """A single cell in a table.

    Attributes:
        row_nums: Row indices occupied by this cell.
        column_nums: Column indices occupied by this cell.
        cell_text: Text content of the cell.
        bbox: Bounding box [x_min, y_min, x_max, y_max] of the cell.
        is_column_header: Whether this cell is a column header.
        is_row_header: Whether this cell is a row header.
    """

    row_nums: list[int]
    column_nums: list[int]
    cell_text: str = ""
    bbox: list[float] | None = None
    is_column_header: bool = False
    is_row_header: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TableCell":
        """Create a TableCell from a cell dict. Extra keys are ignored."""
        return cls(
            row_nums=d["row_nums"],
            column_nums=d["column_nums"],
            cell_text=d.get("cell_text", ""),
            bbox=d.get("bbox"),
            is_column_header=d.get("is_column_header", False),
            is_row_header=d.get("is_row_header", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a cell dict."""
        d: dict[str, Any] = {
            "row_nums": self.row_nums,
            "column_nums": self.column_nums,
            "cell_text": self.cell_text,
            "is_column_header": self.is_column_header,
            "is_row_header": self.is_row_header,
        }
        if self.bbox is not None:
            d["bbox"] = self.bbox
        return d


def html_to_cell_list(table_html: str) -> list[TableCell] | None:
    """Parse an HTML representation of a table into a list of cells.

    Args:
        table_html: HTML string of the table to parse.

    Returns:
        A list of TableCell objects with row_nums, column_nums, is_column_header, and cell_text,
        or None if the HTML cannot be parsed.
    """
    try:
        tree = ET.fromstring(table_html)
    except ET.ParseError:
        return None

    table_cells = []

    occupied_columns_by_row: defaultdict[int, set[int]] = defaultdict(set)
    current_row = -1

    # DFS traversal of the HTML tree to extract cell (td/th) elements
    stack = []
    stack.append((tree, False))
    while len(stack) > 0:
        current, in_header = stack.pop()

        if current.tag == "tr":
            current_row += 1

        if current.tag == "td" or current.tag == "th":
            if "colspan" in current.attrib:
                colspan = int(current.attrib["colspan"])
            else:
                colspan = 1
            if "rowspan" in current.attrib:
                rowspan = int(current.attrib["rowspan"])
            else:
                rowspan = 1
            row_nums = list(range(current_row, current_row + rowspan))
            try:
                max_occupied_column = max(occupied_columns_by_row[current_row])
                current_column = min(
                    set(range(max_occupied_column + 2)).difference(
                        occupied_columns_by_row[current_row]
                    )
                )
            except ValueError:
                current_column = 0
            column_nums = list(range(current_column, current_column + colspan))
            for row_num in row_nums:
                occupied_columns_by_row[row_num].update(column_nums)

            cell = TableCell(
                row_nums=row_nums,
                column_nums=column_nums,
                is_column_header=current.tag == "th" or in_header,
                cell_text=" ".join(current.itertext()),
            )
            table_cells.append(cell)

        children = list(current)
        for child in children[::-1]:
            stack.append((child, in_header or current.tag == "th" or current.tag == "thead"))

    return table_cells


def _cell_list_to_grid(cell_list: list[TableCell], key: str, default_value: Any) -> list[list[Any]]:
    """Convert from a list of TableCells to a matrix of grid cell features.

    This matrix representation is the input to GriTS.

    Args:
        cell_list: List of TableCell objects.
        key: The cell attribute to extract ('bbox' for GriTS_Loc, 'cell_text' for GriTS_Con).
        default_value: Value used to initialize grid cells that are not covered by any cell.

    Returns:
        A 2D list (grid) of cell features indexed by [row][column].
    """
    if len(cell_list) == 0:
        return [[]]
    num_rows = max([max(cell.row_nums) for cell in cell_list]) + 1
    num_columns = max([max(cell.column_nums) for cell in cell_list]) + 1
    cell_grid = [[default_value for _ in range(num_columns)] for _ in range(num_rows)]
    for cell in cell_list:
        for row_num in cell.row_nums:
            for column_num in cell.column_nums:
                cell_grid[row_num][column_num] = getattr(cell, key)

    return cell_grid


def cell_list_to_grid_con(cell_list: list[TableCell]) -> list[list[Any]]:
    """Convert from a list of TableCells to a matrix of cell text, used for computing GriTS_Con.

    Args:
        cell_list: List of TableCell objects with row_nums, column_nums, and cell_text.

    Returns:
        A 2D list (grid) of cell text strings indexed by [row][column].
    """
    default_value = ""
    return _cell_list_to_grid(cell_list, key="cell_text", default_value=default_value)


def cell_list_to_grid_loc(cell_list: list[TableCell]) -> list[list[Any]]:
    """Convert from a list of TableCells to a matrix of cell bounding boxes, used for computing GriTS_Loc.

    Args:
        cell_list: List of TableCell objects with row_nums, column_nums, and bbox.

    Returns:
        A 2D list (grid) of bounding boxes indexed by [row][column].
    """
    default_value = [0.0, 0.0, 0.0, 0.0]
    return _cell_list_to_grid(cell_list, key="bbox", default_value=default_value)


def cell_list_to_grid_top(cell_list: list[TableCell]) -> list[list[Any]]:
    """Convert from a list of TableCells to the matrix/grid of relative spans (boxes), used for computing GriTS_Top.

    For the cell at grid location (i,j), let a(i,j) be its rowspan,
    let β(i,j) be its colspan, let p(i,j) be the minimum row it occupies,
    and let θ(i,j) be the minimum column it occupies. Its relative span is
    bounding box [θ(i,j)-j, p(i,j)-i, θ(i,j)-j+β(i,j), p(i,j)-i+a(i,j)].

    It gives the size and location of the cell each grid cell belongs to
    relative to the current grid cell location, in grid coordinate units.
    Note that for a non-spanning cell this will always be [0, 0, 1, 1].

    Args:
        cell_list: List of TableCell objects with row_nums and column_nums.

    Returns:
        A 2D list (grid) of relative span values indexed by [row][column].
    """
    if len(cell_list) == 0:
        return [[]]
    num_rows = max([max(cell.row_nums) for cell in cell_list]) + 1
    num_columns = max([max(cell.column_nums) for cell in cell_list]) + 1

    # Initialize grid with [0, 0, 1, 1] (non-spanning cells) instead of 0.0
    # This handles cases where HTML tables have missing cells
    default_value = [0, 0, 1, 1]
    cell_grid = [[default_value for _ in range(num_columns)] for _ in range(num_rows)]

    for cell in cell_list:
        min_row_num = min(cell.row_nums)
        min_column_num = min(cell.column_nums)
        max_row_num = max(cell.row_nums) + 1
        max_column_num = max(cell.column_nums) + 1
        for row_num in cell.row_nums:
            for column_num in cell.column_nums:
                cell_grid[row_num][column_num] = [
                    min_column_num - column_num,
                    min_row_num - row_num,
                    max_column_num - column_num,
                    max_row_num - row_num,
                ]

    return cell_grid


def html_to_grids(table_html: str) -> dict[str, list[list[Any]]] | None:
    """Convert an HTML table to con and top grids.

    This is a convenience function that chains html_to_cell_list with
    cell_list_to_grid_con and cell_list_to_grid_top.

    Args:
        table_html: HTML string of the table to convert.

    Returns:
        A dict with keys "con" and "top" mapping to their respective grids,
        or None if the HTML cannot be parsed.
    """
    cell_list = html_to_cell_list(table_html)
    if cell_list is None:
        return None
    return {
        "con": cell_list_to_grid_con(cell_list),
        "top": cell_list_to_grid_top(cell_list),
    }
