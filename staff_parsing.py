import cv2
import numpy as np

from gtrs.debug import Debug
from gtrs.model import TabEncodedSymbol, TabStaff
from gtrs.simple_logging import eprint
from gtrs.staff_dewarping import dewarp_staff
from gtrs.transformer.configs import TabConfig
from gtrs.transformer.staff2score import TabStaff2Score, build_canvas_from_staff
from gtrs.type_definitions import NDArray


def parse_tab_staffs(
    debug: Debug,
    staffs: list[TabStaff],
    image: NDArray,
    config: TabConfig | None = None,
) -> list[tuple[TabStaff, list[TabEncodedSymbol]]]:
    model = TabStaff2Score(config)

    results: list[tuple[TabStaff, list[TabEncodedSymbol]]] = []

    for i, staff in enumerate(staffs):
        eprint(f"Processing staff {i + 1}/{len(staffs)}")

        try:
            dewarped = dewarp_staff(image, staff, debug)
            debug.write_image(f"staff_{i}_dewarped", dewarped)

            staff_image = build_canvas_from_staff(dewarped, staff, config)
            debug.write_image(f"staff_{i}_canvas", staff_image)

            symbols = model.predict(staff_image)

            eprint(f"Staff {i + 1}: {len(symbols)} symbols predicted")

            avg_conf = (
                sum(s.confidence for s in symbols) / len(symbols) if symbols else 0.0
            )
            eprint(f"Staff {i + 1}: avg confidence={avg_conf:.3f}")

            results.append((staff, symbols))

        except Exception as e:
            eprint(f"Error processing staff {i + 1}: {e}")
            results.append((staff, []))

    return results