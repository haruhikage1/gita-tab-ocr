from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from gtrs import constants
from gtrs.bounding_boxes import (
    AngledBoundingBox,
    BoundingBox,
    BoundingEllipse,
    DebugDrawable,
    RotatedBoundingBox,
)
from gtrs.type_definitions import NDArray


class InputPredictions:
    def __init__(
        self,
        original: NDArray,
        preprocessed: NDArray,
        tab_lines: NDArray,
        fret_numbers: NDArray,
        technique_marks: NDArray,
        tab_clef: NDArray,
        rhythm_symbols: NDArray,
        bar_lines: NDArray,
        special_marks: NDArray,
    ) -> None:
        self.original = original
        self.preprocessed = preprocessed
        self.tab_lines = tab_lines
        self.fret_numbers = fret_numbers
        self.technique_marks = technique_marks
        self.tab_clef = tab_clef
        self.rhythm_symbols = rhythm_symbols
        self.bar_lines = bar_lines
        self.special_marks = special_marks


class SymbolOnTabStaff(DebugDrawable):
    def __init__(self, center: tuple[float, float]) -> None:
        self.center = center

    def copy(self) -> SymbolOnTabStaff:
        return SymbolOnTabStaff(self.center)

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> SymbolOnTabStaff:
        copy = self.copy()
        copy.center = transformation(self.center)
        return copy


class TabClef(SymbolOnTabStaff):
    def __init__(self, box: BoundingBox) -> None:
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "TAB",
            (self.box.box[0], self.box.box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "TabClef(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> TabClef:
        return TabClef(self.box)


class BarLine(SymbolOnTabStaff):
    def __init__(self, box: RotatedBoundingBox) -> None:
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)

    def __str__(self) -> str:
        return "BarLine(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> BarLine:
        return BarLine(self.box)


class FretNumber(SymbolOnTabStaff):
    def __init__(self, box: BoundingBox, string_index: int, fret: int) -> None:
        super().__init__(box.center)
        self.box = box
        self.string_index = string_index
        self.fret = fret

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.fret),
            (int(self.box.center[0]), int(self.box.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return f"FretNumber(s{self.string_index + 1}f{self.fret})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> FretNumber:
        return FretNumber(self.box, self.string_index, self.fret)


class TechniqueType(Enum):
    HAMMER_ON = "h"
    PULL_OFF = "p"
    SLIDE = "s"
    BEND = "b"
    RELEASE = "r"
    VIBRATO = "~"
    MUTE = "x"
    NATURAL_HARMONIC = "nh"
    ARTIFICIAL_HARMONIC = "ah"
    TAP = "t"
    SLIDE_UP = "su"
    SLIDE_DOWN = "sd"
    TRILL = "tr"
    LET_RING = "lr"
    PALM_MUTE = "pm"

    def __str__(self) -> str:
        return self.value


class TechniqueMark(SymbolOnTabStaff):
    def __init__(self, box: BoundingBox, technique: TechniqueType) -> None:
        super().__init__(box.center)
        self.box = box
        self.technique = technique

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 255, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            str(self.technique),
            (int(self.box.center[0]), int(self.box.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return f"TechniqueMark({self.technique})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> TechniqueMark:
        return TechniqueMark(self.box, self.technique)


class RhythmSymbol(SymbolOnTabStaff):
    def __init__(self, box: BoundingBox, symbol_type: str) -> None:
        super().__init__(box.center)
        self.box = box
        self.symbol_type = symbol_type

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (0, 0, 255)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            self.symbol_type,
            (int(self.box.center[0]), int(self.box.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return f"RhythmSymbol({self.symbol_type})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> RhythmSymbol:
        return RhythmSymbol(self.box, self.symbol_type)


class TabStaffPoint:
    def __init__(self, x: float, y: list[float], angle: float) -> None:
        if len(y) % constants.TAB_LINES_COUNT != 0:
            raise ValueError(
                f"Tab staff must consist of multiples of {constants.TAB_LINES_COUNT} lines, "
                f"got {len(y)}"
            )
        self.x = x
        self.y = y
        self.angle = angle
        self.average_unit_size = float(np.mean(np.diff(y)))

    def merge(self, other: TabStaffPoint) -> TabStaffPoint:
        if abs(self.x - other.x) > 1e-3:
            raise ValueError("Can't merge points at different positions")
        y: list[float] = []
        y.extend(self.y)
        y.extend(other.y)
        angle = (self.angle + other.angle) / 2
        return TabStaffPoint(self.x, sorted(y), angle)

    def get_string_position(self, string_index: int) -> float:
        if string_index < 0 or string_index >= len(self.y):
            raise IndexError(f"String index {string_index} out of range [0, {len(self.y)})")
        return self.y[string_index]

    def find_string_index(self, y_coord: float) -> int:
        distances = [abs(y_val - y_coord) for y_val in self.y]
        return int(np.argmin(distances))

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> TabStaffPoint:
        xy = [transformation((self.x, y_value)) for y_value in self.y]
        average_x = float(np.mean([x for x, _ in xy]))
        return TabStaffPoint(average_x, [y for _, y in xy], self.angle)

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            [int(self.x), int(self.y[0]), int(self.x), int(self.y[-1])], np.array([]), -2
        )

    def __str__(self) -> str:
        return f"TabStaffPoint(x={self.x}, y_mid={self.y[len(self.y) // 2]})"

    def __repr__(self) -> str:
        return str(self)


class TabStaff(DebugDrawable):
    def __init__(self, grid: list[TabStaffPoint]) -> None:
        self.grid = grid
        self.min_x = grid[0].x
        self.max_x = grid[-1].x
        self.min_y = min(min(p.y) for p in grid)
        self.max_y = max(max(p.y) for p in grid)
        self.average_unit_size = float(np.median([p.average_unit_size for p in grid]))
        self.symbols: list[SymbolOnTabStaff] = []
        self.is_grandstaff = False
        self._y_tolerance = 4 * self.average_unit_size

    def is_on_staff_zone(self, item: AngledBoundingBox) -> bool:
        point = self.get_at(item.center[0])
        if point is None:
            return False
        if (
            item.center[1] > point.y[-1] + self._y_tolerance
            or item.center[1] < point.y[0] - self._y_tolerance
        ):
            return False
        return True

    def merge(self, other: TabStaff) -> TabStaff:
        grid_a: dict[int, TabStaffPoint] = {}
        for p in self.grid:
            grid_a[int(round(p.x))] = p
        grid_b: dict[int, TabStaffPoint] = {}
        for p in other.grid:
            grid_b[int(round(p.x))] = p
        x_positions = set(grid_a.keys()).intersection(grid_b.keys())

        grid = [grid_a[x].merge(grid_b[x]) for x in sorted(x_positions)]
        result = TabStaff(grid)
        result.symbols.extend(self.symbols)
        result.symbols.extend(other.symbols)
        result.is_grandstaff = True
        return result

    def add_symbol(self, symbol: SymbolOnTabStaff) -> None:
        self.symbols.append(symbol)

    def get_at(self, x: float) -> TabStaffPoint | None:
        closest_point = min(self.grid, key=lambda p: abs(p.x - x))
        if abs(closest_point.x - x) > constants.STAFF_POSITION_TOLERANCE:
            return None
        return closest_point

    def y_distance_to(self, point: tuple[float, float]) -> float:
        staff_point = self.get_at(point[0])
        if staff_point is None:
            return 1e10
        return min(abs(y - point[1]) for y in staff_point.y)

    def get_bar_lines(self) -> list[BarLine]:
        return [s for s in self.symbols if isinstance(s, BarLine)]

    def get_clefs(self) -> list[TabClef]:
        return [s for s in self.symbols if isinstance(s, TabClef)]

    def get_fret_numbers(self) -> list[FretNumber]:
        return [s for s in self.symbols if isinstance(s, FretNumber)]

    def get_technique_marks(self) -> list[TechniqueMark]:
        return [s for s in self.symbols if isinstance(s, TechniqueMark)]

    def get_rhythm_symbols(self) -> list[RhythmSymbol]:
        return [s for s in self.symbols if isinstance(s, RhythmSymbol)]

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        if len(self.grid) == 0:
            return
        for i in range(len(self.grid[0].y)):
            for j in range(len(self.grid) - 1):
                p1 = self.grid[j]
                p2 = self.grid[j + 1]
                cv2.line(
                    img,
                    (int(p1.x), int(p1.y[i])),
                    (int(p2.x), int(p2.y[i])),
                    color,
                    thickness=2,
                )

    def extend_to_x_range(self, min_x: int, max_x: int) -> TabStaff:
        grid = self.grid.copy()
        if 0 <= min_x < grid[0].x:
            grid.insert(0, TabStaffPoint(min_x, grid[0].y, grid[0].angle))
        if 0 <= max_x > grid[-1].x:
            grid.append(TabStaffPoint(max_x, grid[-1].y, grid[-1].angle))
        return TabStaff(grid)

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> TabStaff:
        result = TabStaff([point.transform_coordinates(transformation) for point in self.grid])
        result.symbols = [symbol.transform_coordinates(transformation) for symbol in self.symbols]
        result.is_grandstaff = self.is_grandstaff
        return result

    def __str__(self) -> str:
        return "TabStaff(" + str.join(", ", [str(s) for s in self.symbols]) + ")"

    def __repr__(self) -> str:
        return str(self)


class TabMultiStaff(DebugDrawable):
    def __init__(
        self, staffs: list[TabStaff], connections: list[RotatedBoundingBox]
    ) -> None:
        self.staffs = sorted(staffs, key=lambda s: s.min_y)
        self.connections = connections

    def merge(self, other: TabMultiStaff) -> TabMultiStaff:
        unique_staffs: list[TabStaff] = []
        unique_connections: list[RotatedBoundingBox] = []
        for staff in self.staffs + other.staffs:
            if staff not in unique_staffs:
                unique_staffs.append(staff)
        for connection in self.connections + other.connections:
            if connection not in unique_connections:
                unique_connections.append(connection)
        return TabMultiStaff(unique_staffs, unique_connections)

    def _score_brace_with_staff_pair(
        self, symbol: RotatedBoundingBox, upper_staff: TabStaff, lower_staff: TabStaff
    ) -> float:
        unit_size = float(np.median([upper_staff.average_unit_size, lower_staff.average_unit_size]))
        x_distance_threshold = constants.GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR * unit_size
        y_overlap_threshold = constants.GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR * symbol.size[1]

        symbol_min_x = symbol.center[0]
        staff_min_x = min(upper_staff.min_x, lower_staff.min_x)
        x_distance = abs(staff_min_x - symbol_min_x)

        symbol_min_y = symbol.center[1] - symbol.size[1] / 2
        symbol_max_y = symbol.center[1] + symbol.size[1] / 2
        y_overlap = min(symbol_max_y, lower_staff.max_y) - max(symbol_min_y, upper_staff.min_y)

        if (
            x_distance < x_distance_threshold
            and y_overlap > y_overlap_threshold
            and y_overlap > x_distance
        ):
            return y_overlap - x_distance
        return 0.0

    class GrandStaffPair:
        def __init__(self, pair_index: list[int], score: float) -> None:
            self.pair_index = pair_index
            self.score = score

        def get_score(self) -> float:
            return self.score

        def get_index(self) -> list[int]:
            return self.pair_index

        def __repr__(self) -> str:
            return f"staff pair: {self.pair_index} with score: {self.score})"

    def _select_grandstaffs(
        self, brace_dot: list[RotatedBoundingBox]
    ) -> list[TabMultiStaff.GrandStaffPair]:
        pair_scores: list[TabMultiStaff.GrandStaffPair] = []
        for i in range(len(self.staffs) - 1):
            best_score = max(
                self._score_brace_with_staff_pair(symbol, self.staffs[i], self.staffs[i + 1])
                for symbol in brace_dot
            )
            if best_score > 0:
                pair_scores.append(TabMultiStaff.GrandStaffPair([i, i + 1], best_score))

        result: list[TabMultiStaff.GrandStaffPair] = []
        used_staff_index: set[int] = set()
        for pair in sorted(pair_scores, key=lambda p: p.get_score(), reverse=True):
            if any(index in used_staff_index for index in pair.get_index()):
                continue
            result.append(pair)
            used_staff_index.update(pair.get_index())
        return result

    def _merge_selected_pairs(
        self, pairs: list[TabMultiStaff.GrandStaffPair]
    ) -> list[TabStaff]:
        result: list[TabStaff] = []
        i = 0
        while i < len(self.staffs):
            if any(i in pair.get_index() for pair in pairs):
                result.append(self.staffs[i].merge(self.staffs[i + 1]))
                i += 2
                continue
            result.append(self.staffs[i])
            i += 1
        return result

    def create_grandstaffs(self, brace_dot: list[RotatedBoundingBox]) -> TabMultiStaff:
        if len(self.staffs) < 2:
            return self
        pairs = self._select_grandstaffs(brace_dot)
        if not pairs:
            return self
        merged_staffs = self._merge_selected_pairs(pairs)
        return TabMultiStaff(merged_staffs, self.connections)

    def break_apart(self) -> list[TabMultiStaff]:
        return [TabMultiStaff([staff], []) for staff in self.staffs]

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for staff in self.staffs:
            staff.draw_onto_image(img, color)
        for connection in self.connections:
            connection.draw_onto_image(img, color)


@dataclass
class TabEncodedSymbol:
    rhythm: str
    tab: str
    technique: str
    articulation: str
    position: str
    confidence: float = 1.0
    x: float = 0.0
    y: float = 0.0

    def get_string(self) -> int:
        if not self.tab.startswith("s") or "f" not in self.tab:
            return 0
        try:
            return int(self.tab[1 : self.tab.index("f")])
        except (ValueError, IndexError):
            return 0

    def get_fret(self) -> int:
        if "f" not in self.tab:
            return -1
        try:
            return int(self.tab[self.tab.index("f") + 1 :])
        except (ValueError, IndexError):
            return -1

    def is_note(self) -> bool:
        return self.rhythm.startswith("note_") or self.rhythm.startswith("rest_")

    def is_rest(self) -> bool:
        return self.rhythm.startswith("rest_")

    def is_valid(self) -> bool:
        if self.is_note() and not self.is_rest():
            if self.tab == "empty":
                return False
            string = self.get_string()
            fret = self.get_fret()
            if string < 1 or string > constants.MAX_STRING:
                return False
            if fret < 0 or fret > constants.MAX_FRET:
                return False
        return True

    def __str__(self) -> str:
        return f"TabEncodedSymbol(r={self.rhythm}, t={self.tab}, tech={self.technique})"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class PreprocessingConfig:
    target_min_width: int = constants.MIN_IMAGE_WIDTH
    deskew_max_angle: float = constants.DESKEW_MAX_ANGLE
    deskew_step: float = constants.DESKEW_STEP
    median_blur_kernel: int = 3
    autocrop_threshold: int = 10


@dataclass
class PreprocessingResult:
    image: NDArray
    original_size: tuple[int, int]
    was_deskewed: bool = False
    deskew_angle: float = 0.0
    was_resized: bool = False
    scale_factor: float = 1.0


@dataclass
class TabSegmentationResult:
    mask: NDArray
    tab_lines: NDArray = field(init=False)
    fret_numbers: NDArray = field(init=False)
    technique_marks: NDArray = field(init=False)
    tab_clef: NDArray = field(init=False)
    rhythm_symbols: NDArray = field(init=False)
    bar_lines: NDArray = field(init=False)
    special_marks: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.tab_lines = (self.mask == 1).astype(np.uint8)
        self.fret_numbers = (self.mask == 2).astype(np.uint8)
        self.technique_marks = (self.mask == 3).astype(np.uint8)
        self.tab_clef = (self.mask == 4).astype(np.uint8)
        self.rhythm_symbols = (self.mask == 5).astype(np.uint8)
        self.bar_lines = (self.mask == 6).astype(np.uint8)
        self.special_marks = (self.mask == 7).astype(np.uint8)


@dataclass
class GuitarTuning:
    name: str
    strings: list[str]

    @staticmethod
    def from_name(name: str) -> GuitarTuning:
        presets: dict[str, list[str]] = {
            "standard": ["E2", "A2", "D3", "G3", "B3", "E4"],
            "drop_d": ["D2", "A2", "D3", "G3", "B3", "E4"],
            "drop_c": ["C2", "G2", "C3", "F3", "A3", "D4"],
            "open_g": ["D2", "G2", "D3", "G3", "B3", "D4"],
            "open_d": ["D2", "A2", "D3", "F#3", "A3", "D4"],
            "eb": ["Eb2", "Ab2", "Db3", "Gb3", "Bb3", "Eb4"],
        }
        if name not in presets:
            raise ValueError(f"Unknown tuning: {name}. Available: {list(presets.keys())}")
        return GuitarTuning(name=name, strings=presets[name])


@dataclass
class TabNoteOutput:
    string: int
    fret: int
    technique: str = ""
    duration: str = ""
    confidence: float = 1.0
    low_confidence: bool = False


@dataclass
class TabMeasureOutput:
    measure_number: int
    notes: list[TabNoteOutput]
    time_signature: str = ""


@dataclass
class TabStaffOutput:
    tuning: GuitarTuning
    measures: list[TabMeasureOutput]
    average_confidence: float = 1.0


@dataclass
class TabScoreOutput:
    staves: list[TabStaffOutput]
    processing_time: float = 0.0
    total_symbols: int = 0