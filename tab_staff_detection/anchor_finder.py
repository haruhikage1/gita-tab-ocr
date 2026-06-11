import cv2
import numpy as np

from gtrs import constants
from gtrs.bounding_boxes import BoundingBox, RotatedBoundingBox, create_rotated_bounding_boxes
from gtrs.debug import Debug
from gtrs.model import TabClef, TabStaff, TabStaffPoint
from gtrs.simple_logging import eprint
from gtrs.tab_staff_detection import (
    TabLineSegment,
    build_staff_from_tab_lines,
    check_spacing_consistency,
    detect_tab_lines,
)
from gtrs.type_definitions import NDArray


def detect_tab_clef_anchors(
    tab_clef_mask: NDArray,
    debug: Debug,
    min_size: tuple[int, int] = (15, 20),
    max_size: tuple[int, int] = (200, 200),
) -> list[BoundingBox]:
    contours, _ = cv2.findContours(
        tab_clef_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    anchors = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size[0] and h >= min_size[1] and w <= max_size[0] and h <= max_size[1]:
            box = BoundingBox((x, y, x + w, y + h), contour, debug_id=i)
            anchors.append(box)

    eprint(f"Found {len(anchors)} TAB clef anchors")
    return anchors


def detect_bar_line_anchors(
    bar_line_mask: NDArray,
    debug: Debug,
    min_height: int = 20,
) -> list[RotatedBoundingBox]:
    bar_lines = create_rotated_bounding_boxes(
        bar_line_mask.astype(np.uint8),
        skip_merging=True,
        min_size=(1, min_height),
    )
    eprint(f"Found {len(bar_lines)} bar line anchors")
    return bar_lines


class StaffAnchor:
    def __init__(self, x: float, y: float, anchor_type: str) -> None:
        self.x = x
        self.y = y
        self.anchor_type = anchor_type

    def __repr__(self) -> str:
        return f"StaffAnchor({self.anchor_type}, x={self.x:.0f}, y={self.y:.0f})"


def find_staff_anchors(
    tab_clef_mask: NDArray,
    bar_line_mask: NDArray,
    debug: Debug,
) -> list[StaffAnchor]:
    anchors: list[StaffAnchor] = []

    clef_boxes = detect_tab_clef_anchors(tab_clef_mask, debug)
    for box in clef_boxes:
        anchors.append(StaffAnchor(box.center[0], box.center[1], "tab_clef"))

    bar_lines = detect_bar_line_anchors(bar_line_mask, debug)
    for line in bar_lines:
        anchors.append(StaffAnchor(line.center[0], line.center[1], "bar_line"))

    anchors.sort(key=lambda a: a.x)
    eprint(f"Total staff anchors: {len(anchors)}")
    return anchors


def build_tab_staffs(
    tab_line_mask: NDArray,
    tab_clef_mask: NDArray,
    bar_line_mask: NDArray,
    debug: Debug,
) -> list[TabStaff]:
    tab_lines = detect_tab_lines(tab_line_mask, debug)

    if len(tab_lines) == 0:
        eprint("No tab lines detected, cannot build staffs")
        return []

    spacings = check_spacing_consistency(tab_lines)
    if spacings:
        eprint(f"Tab line spacings: {[f'{s:.1f}' for s in spacings]}")

    anchors = find_staff_anchors(tab_clef_mask, bar_line_mask, debug)

    if len(anchors) == 0:
        eprint("No anchors found, building single staff from all tab lines")
        if len(tab_lines) >= constants.TAB_LINES_COUNT:
            staff = build_staff_from_tab_lines(tab_lines[:constants.TAB_LINES_COUNT])
            return [staff]
        return []

    clef_anchors = [a for a in anchors if a.anchor_type == "tab_clef"]
    if len(clef_anchors) == 0:
        clef_anchors = anchors

    staffs = _group_tab_lines_by_anchors(tab_lines, clef_anchors)

    eprint(f"Built {len(staffs)} tab staffs")
    return staffs


def _group_tab_lines_by_anchors(
    tab_lines: list[TabLineSegment],
    anchors: list[StaffAnchor],
) -> list[TabStaff]:
    if len(tab_lines) < constants.TAB_LINES_COUNT:
        eprint(f"Not enough tab lines ({len(tab_lines)}) to form a staff")
        return []

    if len(anchors) <= 1:
        staff = build_staff_from_tab_lines(tab_lines[:constants.TAB_LINES_COUNT])
        return [staff]

    staffs: list[TabStaff] = []
    used_lines: set[int] = set()

    for anchor in sorted(anchors, key=lambda a: a.y):
        candidate_lines = []
        for i, line in enumerate(tab_lines):
            if i not in used_lines:
                candidate_lines.append((i, line))

        if len(candidate_lines) < constants.TAB_LINES_COUNT:
            break

        closest = sorted(candidate_lines, key=lambda il: abs(il[1].center_y - anchor.y))
        selected = closest[:constants.TAB_LINES_COUNT]
        selected_indices = [il[0] for il in selected]
        selected_lines = [il[1] for il in selected]

        staff = build_staff_from_tab_lines(selected_lines)
        staffs.append(staff)
        used_lines.update(selected_indices)

    if not staffs and len(tab_lines) >= constants.TAB_LINES_COUNT:
        staff = build_staff_from_tab_lines(tab_lines[:constants.TAB_LINES_COUNT])
        staffs.append(staff)

    return staffs