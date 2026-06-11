import json
import os
import random

import cv2
import numpy as np

from gtrs.segmentation.categories import NUM_TAB_CATEGORIES, TabCategory
from gtrs.simple_logging import eprint
from gtrs.transformer.configs import TabConfig
from gtrs.type_definitions import NDArray


class SegmentationAnnotator:
    def __init__(self, num_categories: int = NUM_TAB_CATEGORIES) -> None:
        self.num_categories = num_categories

    def validate_mask(self, mask_path: str) -> bool:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            eprint(f"ANNOTATION_FORMAT_ERROR: Cannot read mask {mask_path}")
            return False
        unique = np.unique(mask)
        if len(unique) > self.num_categories:
            eprint(
                f"ANNOTATION_FORMAT_ERROR: Mask has {len(unique)} unique values, "
                f"expected <= {self.num_categories}"
            )
            return False
        for val in unique:
            if val < 0 or val >= self.num_categories:
                eprint(f"ANNOTATION_FORMAT_ERROR: Invalid category value {val}")
                return False
        return True

    def convert_labelme_to_mask(
        self, labelme_json_path: str, output_path: str, image_shape: tuple[int, int]
    ) -> bool:
        try:
            with open(labelme_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            eprint(f"ANNOTATION_FORMAT_ERROR: {e}")
            return False

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            category_id = self._label_to_category(label)
            if category_id < 0:
                continue
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], category_id)

        cv2.imwrite(output_path, mask)
        return True

    @staticmethod
    def _label_to_category(label: str) -> int:
        mapping = {
            "background": 0, "tab_lines": 1, "fret_numbers": 2,
            "technique_marks": 3, "tab_clef": 4, "rhythm_symbols": 5,
            "bar_lines": 6, "special_marks": 7,
        }
        return mapping.get(label, -1)


class SequenceAnnotator:
    def __init__(self, config: TabConfig | None = None) -> None:
        self.config = config or TabConfig()

    def validate_sequence(self, sequence_path: str) -> bool:
        try:
            with open(sequence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            eprint(f"ANNOTATION_FORMAT_ERROR: {e}")
            return False

        for symbol in data.get("symbols", []):
            for branch, vocab in [
                ("rhythm", self.config.rhythm_vocab),
                ("tab", self.config.tab_vocab),
                ("technique", self.config.technique_vocab),
                ("articulation", self.config.articulation_vocab),
                ("position", self.config.position_vocab),
            ]:
                token = symbol.get(branch, "")
                if token and token not in vocab:
                    eprint(f"ANNOTATION_FORMAT_ERROR: Unknown {branch} token '{token}'")
                    return False
        return True


class DatasetSplitter:
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, data_dir: str, output_dir: str) -> dict[str, int]:
        files = [
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        random.seed(self.seed)
        random.shuffle(files)

        n = len(files)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, split_files in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            split_list_path = os.path.join(output_dir, f"{split_name}.txt")
            with open(split_list_path, "w", encoding="utf-8") as f:
                for fname in split_files:
                    f.write(fname + "\n")

        stats = {k: len(v) for k, v in splits.items()}
        eprint(f"Dataset split: {stats}")
        return stats