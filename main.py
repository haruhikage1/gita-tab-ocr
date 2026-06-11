import argparse
import glob
import os
import sys
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from gtrs import download_utils
from gtrs.debug import Debug
from gtrs.model import (
    GuitarTuning,
    PreprocessingConfig,
    PreprocessingResult,
    TabEncodedSymbol,
    TabMeasureOutput,
    TabNoteOutput,
    TabScoreOutput,
    TabStaffOutput,
)
from gtrs.output.ascii_tab_generator import ASCIITabGenerator
from gtrs.output.json_generator import JSONGenerator
from gtrs.output.musicxml_generator import MusicXMLTabGenerator
from gtrs.preprocessing import (
    autocrop,
    binarize,
    convert_to_grayscale,
    deskew,
    remove_noise,
    resize_if_needed,
    validate_format,
)
from gtrs.simple_logging import eprint
from gtrs.staff_parsing import parse_tab_staffs
from gtrs.tab_staff_detection.anchor_finder import build_tab_staffs
from gtrs.tab_staff_detection import (
    make_tab_lines_stronger,
)
from gtrs.transformer.configs import TabConfig
from gtrs.type_definitions import NDArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class GpuSupport(Enum):
    NO = "no"
    AUTO = "auto"
    FORCE = "force"


class OutputFormat(Enum):
    MUSICXML = "musicxml"
    ASCII = "ascii"
    JSON = "json"


class InvalidProgramArgumentException(Exception):
    pass


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    use_gpu_inference: bool
    output_format: OutputFormat
    tuning_name: str
    output_dir: str | None


def preprocess_image(image: NDArray, config: PreprocessingConfig) -> PreprocessingResult:
    original_size = image.shape[:2]

    gray = convert_to_grayscale(image)

    gray, scale_factor = resize_if_needed(gray, config.target_min_width)
    was_resized = scale_factor > 1.0

    gray = remove_noise(gray, config.median_blur_kernel)

    deskewed, angle = deskew(gray, config.deskew_max_angle, config.deskew_step)
    was_deskewed = abs(angle) > 0.1

    result = autocrop(deskewed, config.autocrop_threshold)

    return PreprocessingResult(
        image=result,
        original_size=original_size,
        was_deskewed=was_deskewed,
        deskew_angle=angle,
        was_resized=was_resized,
        scale_factor=scale_factor,
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def process_image(image_path: str, config: ProcessingConfig) -> None:
    eprint("Processing " + image_path)

    try:
        validate_format(image_path)
    except ValueError as e:
        raise InvalidProgramArgumentException(str(e)) from e

    image = cv2.imread(image_path)
    if image is None:
        raise InvalidProgramArgumentException(
            "Failed to read image: " + image_path
        )

    preprocessing_config = PreprocessingConfig()
    preprocessed = preprocess_image(image, preprocessing_config)

    debug = Debug(preprocessed.image, image_path, config.enable_debug)
    debug.write_image("preprocessed", preprocessed.image)

    eprint("Preprocessing complete: "
           f"deskewed={preprocessed.was_deskewed}({preprocessed.deskew_angle:.1f}deg), "
           f"resized={preprocessed.was_resized}({preprocessed.scale_factor:.2f}x)")

    tuning = GuitarTuning.from_name(config.tuning_name)
    eprint(f"Using tuning: {tuning.name} ({', '.join(tuning.strings)})")

    t0 = perf_counter()

    from gtrs.segmentation.inference import extract
    seg_result = extract(
        preprocessed.image, image_path,
        use_cache=config.enable_cache,
        use_gpu_inference=config.use_gpu_inference,
    )

    tab_line_mask = make_tab_lines_stronger(seg_result.tab_lines, (1, 2))
    debug.write_threshold_image("tab_lines", tab_line_mask)
    debug.write_threshold_image("fret_numbers", seg_result.fret_numbers)
    debug.write_threshold_image("technique_marks", seg_result.technique_marks)
    debug.write_threshold_image("tab_clef", seg_result.tab_clef)
    debug.write_threshold_image("rhythm_symbols", seg_result.rhythm_symbols)
    debug.write_threshold_image("bar_lines", seg_result.bar_lines)

    staffs = build_tab_staffs(
        tab_line_mask, seg_result.tab_clef, seg_result.bar_lines, debug
    )
    eprint(f"Found {len(staffs)} tab staffs")

    if len(staffs) == 0:
        raise RuntimeError("No tab staffs found in image")

    transformer_config = TabConfig()
    transformer_config.use_gpu_inference = config.use_gpu_inference

    parsed_staffs = parse_tab_staffs(
        debug, staffs, preprocessed.image, config=transformer_config
    )

    score = _build_score_output(parsed_staffs, tuning)
    score.processing_time = perf_counter() - t0

    eprint(f"Recognition complete: {len(score.staves)} staves, "
           f"{score.total_symbols} symbols, "
           f"{score.processing_time:.2f}s")

    output_dir = config.output_dir or os.path.dirname(image_path)
    base_name = os.path.basename(image_path)

    if config.output_format == OutputFormat.MUSICXML:
        output_path = replace_extension(os.path.join(output_dir, base_name), ".musicxml")
        MusicXMLTabGenerator(tuning).write(output_path, score)
    elif config.output_format == OutputFormat.ASCII:
        output_path = replace_extension(os.path.join(output_dir, base_name), "_tab.txt")
        ASCIITabGenerator().write(output_path, score)
    elif config.output_format == OutputFormat.JSON:
        output_path = replace_extension(os.path.join(output_dir, base_name), ".json")
        JSONGenerator().write(output_path, score)

    debug.clean_debug_files_from_previous_runs()


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_debug = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_debug)


def download_weights(use_gpu_inference: bool) -> None:
    eprint("Model weight download not yet configured for GTRS")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gtrs", description="Guitar Tablature Recognition System"
    )
    parser.add_argument("image", type=str, nargs="?", help="Path to the image or directory to process")
    parser.add_argument(
        "--init", action="store_true",
        help="Downloads the models if they are missing and then exits",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true",
        help="Read an existing cache file or create a new one",
    )
    parser.add_argument(
        "--format", type=OutputFormat, choices=list(OutputFormat), default=OutputFormat.MUSICXML,
        help="Output format (default: musicxml)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--tuning", type=str, default="standard",
        help="Guitar tuning preset: standard, drop_d, drop_c, open_g, open_d, eb",
    )
    parser.add_argument(
        "--gpu", type=GpuSupport, choices=list(GpuSupport), default=GpuSupport.AUTO,
        help="GPU inference mode",
    )

    args = parser.parse_args()

    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()
    use_gpu_inference = (
        args.gpu == GpuSupport.AUTO and has_gpu_support
    ) or args.gpu == GpuSupport.FORCE

    download_weights(use_gpu_inference)
    if args.init:
        eprint("Init finished")
        return

    config = ProcessingConfig(
        enable_debug=args.debug,
        enable_cache=args.cache,
        use_gpu_inference=use_gpu_inference,
        output_format=args.format,
        tuning_name=args.tuning,
        output_dir=args.output,
    )

    if args.debug:
        eprint("Using Log Level 2 for OnnxRuntime")
        ort.set_default_logger_severity(2)
    else:
        ort.set_default_logger_severity(3)

    if not args.image:
        eprint("No image provided")
        parser.print_help()
        sys.exit(1)
    elif os.path.isfile(args.image):
        try:
            process_image(args.image, config)
        except InvalidProgramArgumentException as e:
            eprint(str(e))
            sys.exit(2)
    elif os.path.isdir(args.image):
        image_files = get_all_image_files_in_folder(args.image)
        eprint("Processing", len(image_files), "files")
        error_files = []
        for image_file in image_files:
            try:
                process_image(image_file, config)
                eprint("Finished", image_file)
            except Exception as e:
                eprint(f"Error processing {image_file}: {e}")
                error_files.append(image_file)
        if error_files:
            eprint("Errors occurred while processing:", error_files)
    else:
        eprint(f"{args.image} is not a valid file or directory")
        sys.exit(2)


def _build_score_output(
    parsed_staffs: list[tuple], tuning: GuitarTuning
) -> TabScoreOutput:
    staves = []
    total_symbols = 0
    for staff, symbols in parsed_staffs:
        measures = _group_symbols_into_measures(symbols)
        avg_conf = sum(s.confidence for s in symbols) / len(symbols) if symbols else 0.0
        total_symbols += len(symbols)
        staves.append(TabStaffOutput(
            tuning=tuning,
            measures=measures,
            average_confidence=avg_conf,
        ))
    return TabScoreOutput(staves=staves, total_symbols=total_symbols)


def _group_symbols_into_measures(
    symbols: list[TabEncodedSymbol],
) -> list[TabMeasureOutput]:
    if not symbols:
        return [TabMeasureOutput(measure_number=1, notes=[])]

    measures = []
    current_notes: list[TabNoteOutput] = []
    measure_num = 1

    for symbol in symbols:
        if symbol.rhythm == "barline":
            measures.append(TabMeasureOutput(
                measure_number=measure_num,
                notes=current_notes,
            ))
            measure_num += 1
            current_notes = []
            continue

        if symbol.is_note() and not symbol.is_rest():
            note = TabNoteOutput(
                string=symbol.get_string(),
                fret=symbol.get_fret(),
                technique=symbol.technique if symbol.technique not in (".", "_") else "",
                duration=symbol.rhythm.replace("note_", "") if symbol.rhythm.startswith("note_") else "",
                confidence=symbol.confidence,
                low_confidence=symbol.confidence < 0.3,
            )
            current_notes.append(note)
        elif symbol.is_rest():
            note = TabNoteOutput(
                string=0,
                fret=-1,
                duration=symbol.rhythm.replace("rest_", "") if symbol.rhythm.startswith("rest_") else "",
                confidence=symbol.confidence,
                low_confidence=symbol.confidence < 0.3,
            )
            current_notes.append(note)

    if current_notes:
        measures.append(TabMeasureOutput(
            measure_number=measure_num,
            notes=current_notes,
        ))

    if not measures:
        measures.append(TabMeasureOutput(measure_number=1, notes=[]))

    return measures


if __name__ == "__main__":
    main()