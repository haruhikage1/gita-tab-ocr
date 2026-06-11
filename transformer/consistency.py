import numpy as np

from gtrs import constants
from gtrs.model import TabEncodedSymbol
from gtrs.simple_logging import eprint
from gtrs.transformer.configs import TabConfig


TAB_WHITELIST: set[str] = set()
TECHNIQUE_WHITELIST: set[str] = {
    "hammer_on", "pull_off", "slide", "bend", "release",
    "vibrato", "mute", "natural_harmonic", "artificial_harmonic",
    "tap", "slide_up", "slide_down", "trill",
}


def _init_tab_whitelist(config: TabConfig) -> None:
    global TAB_WHITELIST
    if not TAB_WHITELIST:
        TAB_WHITELIST = {
            k for k in config.tab_vocab.keys()
            if k.startswith("s") and "f" in k
        }
        TAB_WHITELIST.update({"mute", "empty"})


def validate_tab_token(token: str, config: TabConfig) -> str:
    _init_tab_whitelist(config)

    if token in TAB_WHITELIST:
        return token

    if token.startswith("s") and "f" in token:
        try:
            string = int(token[1:token.index("f")])
            fret = int(token[token.index("f") + 1:])

            clamped_string = max(1, min(string, constants.MAX_STRING))
            clamped_fret = max(0, min(fret, constants.MAX_FRET))

            corrected = f"s{clamped_string}f{clamped_fret}"
            if corrected != token:
                eprint(
                    f"WARN_INVALID_TAB_TOKEN: {token} -> {corrected} "
                    f"(string {string}->{clamped_string}, fret {fret}->{clamped_fret})"
                )
            return corrected
        except (ValueError, IndexError):
            pass

    eprint(f"WARN_INVALID_TAB_TOKEN: {token} -> empty (unparseable)")
    return "empty"


def validate_technique_token(token: str) -> str:
    if token in TECHNIQUE_WHITELIST or token in (".", "_"):
        return token
    eprint(f"WARN_INVALID_TECHNIQUE_TOKEN: {token} -> .")
    return "."


def check_rhythm_tab_consistency(symbol: TabEncodedSymbol) -> TabEncodedSymbol:
    if symbol.is_note() and not symbol.is_rest():
        if symbol.tab == "empty" or symbol.tab == "_":
            eprint(
                f"WARN_INCONSISTENT: Rhythm={symbol.rhythm} is note but Tab={symbol.tab}"
            )
    return symbol


def apply_consistency_checks(
    symbols: list[TabEncodedSymbol], config: TabConfig
) -> list[TabEncodedSymbol]:
    result = []
    for symbol in symbols:
        corrected_tab = validate_tab_token(symbol.tab, config)
        corrected_technique = validate_technique_token(symbol.technique)

        corrected = TabEncodedSymbol(
            rhythm=symbol.rhythm,
            tab=corrected_tab,
            technique=corrected_technique,
            articulation=symbol.articulation,
            position=symbol.position,
            confidence=symbol.confidence,
            x=symbol.x,
            y=symbol.y,
        )
        corrected = check_rhythm_tab_consistency(corrected)
        result.append(corrected)

    return result


def compute_consistency_loss(
    rhythm_logits: np.ndarray,
    tab_logits: np.ndarray,
    technique_logits: np.ndarray,
    articulation_logits: np.ndarray,
    position_logits: np.ndarray,
) -> float:
    rhythm_pred = rhythm_logits.argmax(axis=-1)
    tab_pred = tab_logits.argmax(axis=-1)

    is_note = (rhythm_pred > 2).astype(np.float32)
    is_empty_tab = (tab_pred <= 4).astype(np.float32)

    inconsistency = is_note * is_empty_tab
    return float(inconsistency.mean())