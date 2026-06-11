import cv2
import numpy as np

from gtrs.type_definitions import NDArray

ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def validate_format(filepath: str) -> str:
    import os

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {ext}. Allowed: {sorted(ALLOWED_FORMATS)}"
        )
    return filepath


def convert_to_grayscale(image: NDArray) -> NDArray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(image: NDArray) -> NDArray:
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


def resize_if_needed(image: NDArray, min_width: int = 1000) -> tuple[NDArray, float]:
    h, w = image.shape[:2]
    if w < min_width:
        scale = min_width / w
        new_w = min_width
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return image, scale
    return image, 1.0


def deskew(image: NDArray, max_angle: float = 5.0, step: float = 0.5) -> tuple[NDArray, float]:
    binary = cv2.bitwise_not(image)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 2:
        return image, 0.0

    best_angle = 0.0
    best_var = -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (image.shape[1], image.shape[0]))
        row_sums = np.sum(rotated, axis=1)
        var = float(np.var(row_sums))
        if var > best_var:
            best_var = var
            best_angle = angle

    if abs(best_angle) < step:
        return image, 0.0

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), best_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=255)
    return deskewed, best_angle


def remove_noise(image: NDArray, kernel_size: int = 3) -> NDArray:
    return cv2.medianBlur(image, kernel_size)


def autocrop(image: NDArray, threshold: int = 10) -> NDArray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return image[y_min : y_max + 1, x_min : x_max + 1]