import cv2
import numpy as np

from gtrs import constants
from gtrs.model import TabEncodedSymbol, TabStaff
from gtrs.simple_logging import eprint
from gtrs.transformer.configs import TabConfig, default_tab_config
from gtrs.transformer.consistency import apply_consistency_checks
from gtrs.transformer.decoder_inference import get_tab_decoder
from gtrs.transformer.encoder_inference import TabEncoder
from gtrs.type_definitions import NDArray


class TabStaff2Score:
    def __init__(self, config: TabConfig | None = None) -> None:
        self.config = config or default_tab_config
        self.encoder = TabEncoder(self.config)
        self.decoder = get_tab_decoder(self.config)

    def predict(self, staff_image: NDArray) -> list[TabEncodedSymbol]:
        canvas = self._prepare_canvas(staff_image)

        context = self.encoder.generate(canvas)

        start_tokens = np.full((1, 1), self.config.bos_token, dtype=np.int64)
        nonote_tokens = np.full((1, 1), self.config.nonote_token, dtype=np.int64)

        symbols = self.decoder.generate(
            start_tokens, nonote_tokens, context=context
        )

        symbols = apply_consistency_checks(symbols, self.config)

        eprint(f"TabStaff2Score: predicted {len(symbols)} symbols")
        return symbols

    def _prepare_canvas(self, staff_image: NDArray) -> NDArray:
        if len(staff_image.shape) == 3:
            staff_image = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)

        h, w = staff_image.shape
        target_h = self.config.max_height
        target_w = self.config.max_width

        canvas = np.full((target_h, target_w), 255, dtype=np.uint8)

        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(staff_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        y_offset = (target_h - new_h) // 2
        x_offset = 0

        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        canvas = canvas.astype(np.float32) / 255.0
        canvas = canvas[np.newaxis, np.newaxis, :, :]

        return canvas


def build_canvas_from_staff(
    image: NDArray,
    staff: TabStaff,
    config: TabConfig | None = None,
) -> NDArray:
    cfg = config or default_tab_config

    x_min = int(max(0, staff.min_x))
    x_max = int(min(image.shape[1], staff.max_x))
    y_min = int(max(0, staff.min_y - 2 * staff.average_unit_size))
    y_max = int(min(image.shape[0], staff.max_y + 2 * staff.average_unit_size))

    staff_image = image[y_min:y_max, x_min:x_max]

    if len(staff_image.shape) == 3:
        staff_image = cv2.cvtColor(staff_image, cv2.COLOR_BGR2GRAY)

    return staff_image