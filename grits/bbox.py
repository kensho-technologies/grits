# Copyright 2025-present Kensho Technologies, LLC.
import dataclasses
from typing import Protocol, Tuple, Union


Number = Union[int, float]


# (x0, y0, x1, y1)
BBOX_COORDINATES = Tuple[Number, Number, Number, Number]


class BBoxProtocol(Protocol):
    """Bounding box protocol.

    The bounding box protocol contains the following protocol variables: x0, y0, x1, y1.
    In a 2D coordinate with the left-to-right x-axis and the top-to-bottom y-axis,
    the (x0, y0) point is the top-left corner of the bounding box and the (x1, y1) point is the
    bottom-right corner of the bounding box.

    This protocol is defined to support polymorphic functions that do not assume any particular
    representation of the bounding box except that it must have four coordinates x0, y0, x1, y1.
    See PEP 544 for more details about Protocols and static duck typing.
    """

    x0: Number
    y0: Number
    x1: Number
    y1: Number


@dataclasses.dataclass(frozen=True, slots=True)
class BBox(BBoxProtocol):
    """Bounding box.

    The bounding box is defined by four numbers (x0, y0, x1, y1).
    In a 2D coordinate with the left-to-right x-axis and the top-to-bottom y-axis,
    the (x0, y0) point is the top-left corner of the bounding box and the (x1, y1) point is the
    bottom-right corner of the bounding box.

    This representation of a bounding box assumes that x0 <= x1 and y0 <= y1.
    """

    x0: Number
    y0: Number
    x1: Number
    y1: Number

    def __post_init__(self) -> None:
        """Validate the bbox coordinates."""
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) is greater than x1 ({self.x1}).")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) is greater than y1 ({self.y1}).")

    @property
    def coordinates(self) -> BBOX_COORDINATES:
        """Standard representation of BBox's coordinates"""
        return self.x0, self.y0, self.x1, self.y1

    @property
    def width(self) -> Number:
        """The width of the BBox."""
        return self.x1 - self.x0

    @property
    def height(self) -> Number:
        """The height of the BBox."""
        return self.y1 - self.y0

    @property
    def area(self) -> Number:
        """The area of the BBox."""
        return self.width * self.height


def is_horizontal_overlap(source: BBoxProtocol, target: BBoxProtocol) -> bool:
    """Is there a horizontal overlap between the source and the target."""
    return source.x0 <= target.x1 and target.x0 <= source.x1


def get_horizontal_overlap(source: BBoxProtocol, target: BBoxProtocol) -> Number:
    """Get the horizontal overlap distance between the two boxes.

    The ordering of the source and the target doesn't matter, as this returns an absolute value.
    """
    if not is_horizontal_overlap(source, target):
        return 0
    else:
        return min(source.x1, target.x1) - max(source.x0, target.x0)


def is_vertical_overlap(source: BBoxProtocol, target: BBoxProtocol) -> bool:
    """Is there a vertical overlap between the source and the target."""
    return source.y0 <= target.y1 and target.y0 <= source.y1


def get_vertical_overlap(source: BBoxProtocol, target: BBoxProtocol) -> Number:
    """Get the vertical overlap distance between the two boxes.

    The ordering of the source and the target doesn't matter, as this returns an absolute value.
    """
    if not is_vertical_overlap(source, target):
        return 0
    else:
        return min(source.y1, target.y1) - max(source.y0, target.y0)


def get_overlap_area(source: BBoxProtocol, target: BBoxProtocol) -> Number:
    """Calculate the overlap area between the source and the target."""
    return get_horizontal_overlap(source, target) * get_vertical_overlap(source, target)
