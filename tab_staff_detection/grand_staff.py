import cv2
import numpy as np

from gtrs import constants
from gtrs.bounding_boxes import RotatedBoundingBox, create_rotated_bounding_boxes
from gtrs.debug import Debug
from gtrs.model import TabMultiStaff, TabStaff
from gtrs.simple_logging import eprint
from gtrs.type_definitions import NDArray


def prepare_brace_image(symbols: NDArray, tab_lines: NDArray) -> NDArray:
    brace_dot = cv2.subtract(symbols, tab_lines)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    out = cv2.erode(brace_dot.astype(np.uint8), kernel_erode)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 35))
    return cv2.dilate(out, kernel_dilate)


def _filter_for_tall_elements(
    brace_dot: list[RotatedBoundingBox], staffs: list[TabStaff]
) -> list[RotatedBoundingBox]:
    rough_unit_size = staffs[0].average_unit_size
    filtered = [
        symbol
        for symbol in brace_dot
        if symbol.size[1] > constants.min_height_for_brace_rough(rough_unit_size)
        and symbol.size[0] < constants.max_width_for_brace_rough(rough_unit_size)
    ]
    result = []
    for symbol in filtered:
        closest_staff = min(staffs, key=lambda staff: staff.y_distance_to(symbol.center))
        unit_size = closest_staff.average_unit_size
        if symbol.size[1] > constants.min_height_for_brace(unit_size):
            result.append(symbol)
    return result


def _get_connections_between_staffs_at_bar_lines(
    staff1: TabStaff, staff2: TabStaff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    bar_lines1 = staff1.get_bar_lines()
    bar_lines2 = staff2.get_bar_lines()
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        symbol_thicker = symbol.make_box_thicker(30)
        overlap1 = [line for line in bar_lines1 if symbol_thicker.is_overlapping(line.box)]
        overlap2 = [line for line in bar_lines2 if symbol_thicker.is_overlapping(line.box)]
        if len(overlap1) >= 1 and len(overlap2) >= 1:
            result.append(symbol)
    return result


def _get_connections_between_staffs_at_clefs(
    staff1: TabStaff, staff2: TabStaff, brace_dot: list[RotatedBoundingBox]
) -> list[RotatedBoundingBox]:
    clefs1 = staff1.get_clefs()
    clefs2 = staff2.get_clefs()
    result: list[RotatedBoundingBox] = []
    for symbol in brace_dot:
        symbol_thicker = symbol.make_box_thicker(
            int(constants.tolerance_for_tab_line_detection(staff1.average_unit_size))
        )
        overlap1 = [clef for clef in clefs1 if symbol_thicker.is_overlapping(clef.box)]
        overlap2 = [clef for clef in clefs2 if symbol_thicker.is_overlapping(clef.box)]
        if len(overlap1) >= 1 and len(overlap2) >= 1:
            result.append(symbol)
    return result


def find_braces_brackets_and_grand_staff_lines(
    debug: Debug,
    staffs: list[TabStaff],
    brace_dot: list[RotatedBoundingBox],
) -> list[TabMultiStaff]:
    if len(staffs) == 0:
        return []

    tall_brace_dot = _filter_for_tall_elements(brace_dot, staffs)

    multi_staffs = [TabMultiStaff([staff], []) for staff in staffs]

    if len(multi_staffs) < 2:
        return multi_staffs

    result = TabMultiStaff(staffs, [])
    result = result.create_grandstaffs(tall_brace_dot)

    return result.break_apart()


def sort_staffs_by_reading_order(staffs: list[TabStaff]) -> list[TabStaff]:
    return sorted(staffs, key=lambda s: (s.min_y, s.min_x))