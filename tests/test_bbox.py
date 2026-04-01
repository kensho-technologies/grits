# Copyright 2025-present Kensho Technologies, LLC.
import pytest

from grits.bbox import (
    BBox,
    get_horizontal_overlap,
    get_overlap_area,
    get_vertical_overlap,
    is_horizontal_overlap,
    is_vertical_overlap,
)


class TestBBox:
    def test_construction_and_properties(self):
        bbox = BBox(1, 2, 5, 8)
        assert bbox.x0 == 1
        assert bbox.y0 == 2
        assert bbox.x1 == 5
        assert bbox.y1 == 8
        assert bbox.coordinates == (1, 2, 5, 8)
        assert bbox.width == 4
        assert bbox.height == 6
        assert bbox.area == 24

    def test_invalid_x_raises(self):
        with pytest.raises(ValueError, match="x0"):
            BBox(10, 0, 5, 10)

    def test_invalid_y_raises(self):
        with pytest.raises(ValueError, match="y0"):
            BBox(0, 10, 5, 5)

    def test_zero_area_point(self):
        bbox = BBox(3, 3, 3, 3)
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0

    def test_zero_area_line(self):
        bbox = BBox(0, 0, 10, 0)
        assert bbox.width == 10
        assert bbox.height == 0
        assert bbox.area == 0

    def test_frozen(self):
        bbox = BBox(0, 0, 1, 1)
        with pytest.raises(AttributeError):
            bbox.x0 = 5


class TestOverlapFunctions:
    def test_horizontal_overlap_true(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 0, 15, 10)
        assert is_horizontal_overlap(a, b) is True

    def test_horizontal_overlap_false(self):
        a = BBox(0, 0, 5, 10)
        b = BBox(6, 0, 10, 10)
        assert is_horizontal_overlap(a, b) is False

    def test_horizontal_overlap_touching_edges(self):
        a = BBox(0, 0, 5, 10)
        b = BBox(5, 0, 10, 10)
        assert is_horizontal_overlap(a, b) is True

    def test_vertical_overlap_true(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(0, 5, 10, 15)
        assert is_vertical_overlap(a, b) is True

    def test_vertical_overlap_false(self):
        a = BBox(0, 0, 10, 5)
        b = BBox(0, 6, 10, 10)
        assert is_vertical_overlap(a, b) is False

    def test_vertical_overlap_touching_edges(self):
        a = BBox(0, 0, 10, 5)
        b = BBox(0, 5, 10, 10)
        assert is_vertical_overlap(a, b) is True

    def test_get_horizontal_overlap_distance(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(6, 0, 15, 10)
        assert get_horizontal_overlap(a, b) == 4

    def test_get_horizontal_overlap_no_overlap(self):
        a = BBox(0, 0, 5, 10)
        b = BBox(6, 0, 10, 10)
        assert get_horizontal_overlap(a, b) == 0

    def test_get_vertical_overlap_distance(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(0, 7, 10, 15)
        assert get_vertical_overlap(a, b) == 3

    def test_get_vertical_overlap_no_overlap(self):
        a = BBox(0, 0, 10, 5)
        b = BBox(0, 6, 10, 10)
        assert get_vertical_overlap(a, b) == 0

    def test_overlap_area_partial(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        assert get_overlap_area(a, b) == 25

    def test_overlap_area_no_overlap(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(6, 6, 10, 10)
        assert get_overlap_area(a, b) == 0

    def test_overlap_area_full_containment(self):
        outer = BBox(0, 0, 20, 20)
        inner = BBox(5, 5, 10, 10)
        assert get_overlap_area(outer, inner) == 25

    def test_overlap_symmetry(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        assert get_overlap_area(a, b) == get_overlap_area(b, a)
