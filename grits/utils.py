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
