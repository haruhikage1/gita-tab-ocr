TAB_LINES_COUNT = 6

TAB_CATEGORIES = 8

MAX_FRET = 24

MAX_STRING = 6

MIN_IMAGE_WIDTH = 1000

DESKEW_MAX_ANGLE = 5.0

DESKEW_STEP = 0.5

SEGMENTATION_STEP_SIZE = 240

SEGMENTATION_NUM_CLASSES = 8

TRANSFORMER_MAX_SEQ_LEN = 608

TRANSFORMER_CANVAS_HEIGHT = 256

TRANSFORMER_CANVAS_WIDTH = 1280

TRANSFORMER_ENCODER_DIM = 512

TRANSFORMER_DECODER_DEPTH = 8

TRANSFORMER_DECODER_HEADS = 8

LOW_CONFIDENCE_THRESHOLD = 0.3

CONSISTENCY_LOSS_GAMMA = 0.1

CRAWLER_MIN_INTERVAL = 1.0

CRAWLER_MAX_RETRIES = 3

CRAWLER_RETRY_DELAY = 60.0

DATASET_TRAIN_RATIO = 0.8

DATASET_VAL_RATIO = 0.1

DATASET_TEST_RATIO = 0.1

API_MAX_FILE_SIZE_MB = 50

API_TIMEOUT_SECONDS = 60


def tolerance_for_tab_line_detection(unit_size: float) -> float:
    return unit_size / 3


def max_line_gap_size(unit_size: float) -> float:
    return 5 * unit_size


def bar_line_max_width(unit_size: float) -> float:
    return 2 * unit_size


def bar_line_min_height(unit_size: float) -> float:
    return 3 * unit_size


def is_short_line(unit_size: float) -> float:
    return unit_size / 5


def min_height_for_brace(unit_size: float) -> float:
    return 4 * unit_size


def max_width_for_brace(unit_size: float) -> float:
    return 3 * unit_size


STAFF_POSITION_TOLERANCE = 50

MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL = 10

GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR = 5

GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR = 0.5

SPACING_INCONSISTENCY_THRESHOLD = 0.2