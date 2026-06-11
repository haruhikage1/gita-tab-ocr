from __future__ import annotations

import cv2
import numpy as np

from gtrs import constants
from gtrs.bounding_boxes import RotatedBoundingBox, create_rotated_bounding_boxes
from gtrs.debug import Debug
from gtrs.model import TabStaff, TabStaffPoint
from gtrs.simple_logging import eprint
from gtrs.type_definitions import NDArray


def prepare_tab_line_image(img: NDArray) -> NDArray:
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    out = cv2.erode(img.astype(np.uint8), kernel_erode)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    return cv2.dilate(out, kernel_dilate)


def make_tab_lines_stronger(img: NDArray, kernel_size: tuple[int, int] = (1, 2)) -> NDArray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img = cv2.dilate(img.astype(np.uint8), kernel)
    img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img


class TabLineSegment:
    def __init__(self, debug_id: int, fragments: list[RotatedBoundingBox]) -> None:
        self.debug_id = debug_id
        self.fragments = sorted(fragments, key=lambda box: box.box[0][0])
        self.min_x = min(line.center[0] - line.size[0] / 2 for line in fragments)
        self.max_x = max(line.center[0] + line.size[0] / 2 for line in fragments)
        self.min_y = min(line.center[1] - line.size[1] / 2 for line in fragments)
        self.max_y = max(line.center[1] + line.size[1] / 2 for line in fragments)

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    def is_overlapping(self, other: TabLineSegment) -> bool:
        for frag in self.fragments:
            for other_frag in other.fragments:
                if frag.is_overlapping(other_frag):
                    return True
        return False

    def __repr__(self) -> str:
        return f"TabLineSegment(id={self.debug_id}, y={self.center_y:.1f})"


def detect_tab_lines(
    tab_line_mask: NDArray,
    debug: Debug,
    min_line_length: int = 50,
) -> list[TabLineSegment]:
    eprint("Detecting tab lines from mask")

    fragments = create_rotated_bounding_boxes(
        tab_line_mask,
        skip_merging=True,
        min_size=(min_line_length, 1),
        max_size=(10000, 100),
    )
    eprint(f"Found {len(fragments)} tab line fragments")

    if len(fragments) == 0:
        return []

    groups: list[list[RotatedBoundingBox]] = []
    for frag in fragments:
        merged = False
        for group in groups:
            for existing in group:
                if frag.is_overlapping_extrapolated(
                    existing, unit_size=10.0
                ):
                    group.append(frag)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            groups.append([frag])

    tab_lines = [TabLineSegment(i, group) for i, group in enumerate(groups)]
    tab_lines.sort(key=lambda line: line.center_y)

    eprint(f"Grouped into {len(tab_lines)} tab lines")

    if len(tab_lines) != constants.TAB_LINES_COUNT:
        eprint(
            f"WARN_TAB_LINE_COUNT_MISMATCH: Expected {constants.TAB_LINES_COUNT} lines, "
            f"found {len(tab_lines)}"
        )

    return tab_lines


def check_spacing_consistency(tab_lines: list[TabLineSegment]) -> list[float]:
    if len(tab_lines) < 2:
        return []

    y_positions = [line.center_y for line in tab_lines]
    spacings = [y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)]
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)

    if std_spacing / mean_spacing > constants.SPACING_INCONSISTENCY_THRESHOLD:
        eprint(
            f"WARN_INCONSISTENT_SPACING: std/mean={std_spacing / mean_spacing:.2f} > "
            f"{constants.SPACING_INCONSISTENCY_THRESHOLD}"
        )

    return spacings


def build_staff_from_tab_lines(
    tab_lines: list[TabLineSegment],
    x_positions: int = 50,
) -> TabStaff:
    y_coords = [line.center_y for line in tab_lines]
    avg_angle = 0.0

    grid = []
    min_x = min(line.min_x for line in tab_lines)
    max_x = max(line.max_x for line in tab_lines)

    step = max(1, int((max_x - min_x) / x_positions))
    for x in np.arange(min_x, max_x, step):
        point = TabStaffPoint(float(x), y_coords, avg_angle)
        grid.append(point)

    if len(grid) == 0:
        mid_x = (min_x + max_x) / 2
        grid.append(TabStaffPoint(mid_x, y_coords, avg_angle))

    return TabStaff(grid)