from enum import IntEnum


class TabCategory(IntEnum):
    BACKGROUND = 0
    TAB_LINES = 1
    FRET_NUMBERS = 2
    TECHNIQUE_MARKS = 3
    TAB_CLEF = 4
    RHYTHM_SYMBOLS = 5
    BAR_LINES = 6
    SPECIAL_MARKS = 7


TAB_CATEGORY_NAMES = {
    TabCategory.BACKGROUND: "background",
    TabCategory.TAB_LINES: "tab_lines",
    TabCategory.FRET_NUMBERS: "fret_numbers",
    TabCategory.TECHNIQUE_MARKS: "technique_marks",
    TabCategory.TAB_CLEF: "tab_clef",
    TabCategory.RHYTHM_SYMBOLS: "rhythm_symbols",
    TabCategory.BAR_LINES: "bar_lines",
    TabCategory.SPECIAL_MARKS: "special_marks",
}

HOMR_TO_TAB_MAPPING = {
    0: TabCategory.BACKGROUND,
    1: TabCategory.BAR_LINES,
    2: TabCategory.FRET_NUMBERS,
    3: TabCategory.TECHNIQUE_MARKS,
    4: TabCategory.TAB_CLEF,
    5: TabCategory.RHYTHM_SYMBOLS,
}

NUM_TAB_CATEGORIES = len(TabCategory)