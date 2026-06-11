import os

import cv2
import numpy as np

from gtrs.simple_logging import eprint
from gtrs.type_definitions import NDArray


class ImageDeduplicator:
    def __init__(self, hash_size: int = 8, similarity_threshold: int = 10) -> None:
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold

    def compute_phash(self, image_path: str) -> str | None:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            img = cv2.resize(img, (self.hash_size + 1, self.hash_size))
            diff = img[:, 1:] > img[:, :-1]
            return "".join(str(int(b)) for row in diff for b in row)
        except Exception:
            return None

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def deduplicate(self, image_dir: str) -> dict[str, int]:
        files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        eprint(f"Computing hashes for {len(files)} images...")

        hashes: dict[str, str] = {}
        for f in files:
            filepath = os.path.join(image_dir, f)
            h = self.compute_phash(filepath)
            if h is not None:
                hashes[f] = h

        duplicates: set[str] = set()
        file_list = list(hashes.keys())
        for i in range(len(file_list)):
            for j in range(i + 1, len(file_list)):
                f1, f2 = file_list[i], file_list[j]
                if f1 in duplicates or f2 in duplicates:
                    continue
                dist = self.hamming_distance(hashes[f1], hashes[f2])
                if dist < self.similarity_threshold:
                    duplicates.add(f2)

        for dup in duplicates:
            os.remove(os.path.join(image_dir, dup))

        stats = {"total": len(files), "duplicates_removed": len(duplicates), "remaining": len(files) - len(duplicates)}
        eprint(f"Deduplication complete: {stats}")
        return stats