import itertools
from typing import Iterable

from gtrs import constants
from gtrs.simple_logging import eprint

nonote = "."
empty = "_"


def build_dict(tokens: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if t in result:
            raise ValueError("Duplicated entry for " + t)
        if not t.strip():
            raise ValueError("Tokens must not be whitespace")
        if t.strip() != t:
            raise ValueError("Tokens must not contain whitespace")
        result[t] = i
    return result


def build_tab_rhythm() -> dict[str, int]:
    rhythm = []

    rhythm.extend(["PAD", "BOS", "EOS"])
    rhythm.append("chord")

    rhythm.extend(["barline", "doublebarline", "bolddoublebarline"])
    rhythm.extend(["repeatStart", "repeatEnd", "repeatEndStart"])
    rhythm.extend(["voltaStart", "voltaStop", "voltaDiscontinue"])

    rhythm.append("clef_TAB")

    rhythm.extend([f"keySignature_{c}" for c in range(-7, 8)])
    rhythm.extend([f"timeSignature/{c}" for c in [1, 2, 3, 4, 6, 8, 12, 16, 32, 48]])

    rhythm.extend([f"rest_{c}m" for c in range(2, 11)])
    kern_base_durations = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128]
    dots = ["", ".", ".."]
    grace = ["", "G"]
    kern_values = [
        f"{d}{g}{dot}" for d, g, dot in itertools.product(kern_base_durations, grace, dots)
    ]
    irregular_durations = [7, 11, 13, 18, 20, 21, 22, 24, 26, 28, 30, 34, 36, 40, 48, 56, 96]

    rhythm.extend([f"note_{d}" for d in kern_values])
    rhythm.extend([f"note_{d}" for d in irregular_durations])
    rhythm.extend([f"rest_{d}" for d in kern_values])
    rhythm.extend([f"rest_{d}" for d in irregular_durations])

    rhythm.extend(["tieSlur"])

    return build_dict(rhythm)


def build_tab() -> dict[str, int]:
    tab = ["PAD", "BOS", "EOS", "mute", "empty"]

    for string in range(1, constants.MAX_STRING + 1):
        for fret in range(0, constants.MAX_FRET + 1):
            tab.append(f"s{string}f{fret}")

    return build_dict(tab)


def build_technique() -> dict[str, int]:
    techniques = [
        nonote, empty,
        "hammer_on", "pull_off", "slide", "bend", "release",
        "vibrato", "mute", "natural_harmonic", "artificial_harmonic",
        "tap", "slide_up", "slide_down", "trill",
    ]
    return build_dict(techniques)


def build_tab_articulation() -> dict[str, int]:
    articulations = [
        nonote, empty,
        "staccato", "accent", "fermata", "arpeggiate",
        "let_ring", "palm_mute",
        "slurStart", "slurStop", "tieStart", "tieStop",
        "tenuto", "breathMark", "tremolo",
    ]
    return build_dict(articulations)


def build_tab_position() -> dict[str, int]:
    positions = [nonote, "upper", "lower"]
    return build_dict(positions)


class TabVocabulary:
    def __init__(self) -> None:
        self.rhythm = build_tab_rhythm()
        self.tab = build_tab()
        self.technique = build_technique()
        self.articulation = build_tab_articulation()
        self.position = build_tab_position()

        eprint(
            f"TabVocabulary sizes: "
            f"rhythm={len(self.rhythm)}, tab={len(self.tab)}, "
            f"technique={len(self.technique)}, "
            f"articulation={len(self.articulation)}, "
            f"position={len(self.position)}"
        )