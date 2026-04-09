# Copyright 2025-present Kensho Technologies, LLC.
from collections import defaultdict
from typing import Any
import xml.etree.ElementTree as ET


def html_to_cell_dicts(table_html: str) -> list[dict[str, Any]] | None:
    """Parse an HTML representation of a table into a list of cells.

    Args:
        table_html: HTML string of the table to parse.

    Returns:
        A list of cell dicts with 'row_nums', 'column_nums', 'is_column_header', and 'cell_text',
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

            cell_dict: dict[str, Any] = {}
            cell_dict["row_nums"] = row_nums
            cell_dict["column_nums"] = column_nums
            cell_dict["is_column_header"] = current.tag == "th" or in_header
            cell_dict["cell_text"] = " ".join(current.itertext())
            table_cells.append(cell_dict)

        children = list(current)
        for child in children[::-1]:
            stack.append((child, in_header or current.tag == "th" or current.tag == "thead"))

    return table_cells
