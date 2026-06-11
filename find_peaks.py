import numpy as np

from gtrs.type_definitions import NDArray


def find_peaks(
    x: NDArray,
    height: float | None = None,
    distance: float | None = None,
    prominence: float | None = None,
) -> tuple[NDArray, dict]:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) < 3:
        return np.array([], dtype=int), {}

    peaks_list: list[int] = []
    i = 1
    while i < len(x) - 1:
        if x[i] > x[i - 1]:
            j = i
            while j < len(x) - 1 and x[j] == x[j + 1]:
                j += 1
            if j < len(x) - 1 and x[j] > x[j + 1]:
                peak_idx = (i + j) // 2
                peaks_list.append(peak_idx)
                i = j + 1
            else:
                i = j + 1
        elif x[i] == x[i - 1]:
            j = i
            while j < len(x) - 1 and x[j] == x[j + 1]:
                j += 1
            if j < len(x) - 1 and x[j] > x[j + 1]:
                peak_idx = (i + j) // 2
                peaks_list.append(peak_idx)
            i = j + 1
        else:
            i += 1

    peaks = np.array(peaks_list, dtype=int)
    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    if height is not None:
        mask = x[peaks] >= height
        peaks = peaks[mask]

    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    if prominence is not None:
        valid_peaks: list[int] = []
        for peak_val in peaks:
            peak = int(peak_val)
            left_min = x[peak]
            for k in range(peak - 1, -1, -1):
                if x[k] > x[peak]:
                    break
                left_min = min(left_min, x[k])
            right_min = x[peak]
            for k in range(peak + 1, len(x)):
                if x[k] > x[peak]:
                    break
                right_min = min(right_min, x[k])
            peak_prominence = x[peak] - max(left_min, right_min)
            if peak_prominence >= prominence:
                valid_peaks.append(peak)
        peaks = np.array(valid_peaks, dtype=int)

    if len(peaks) == 0:
        return np.array([], dtype=int), {}

    if distance is not None and len(peaks) > 1:
        sorted_indices = np.argsort(x[peaks])[::-1]
        sorted_peaks = peaks[sorted_indices]
        keep: list[int] = []
        for peak_val in sorted_peaks:
            peak = int(peak_val)
            if len(keep) == 0 or all(abs(k - peak) >= distance for k in keep):
                keep.append(peak)
        peaks = np.array(sorted(keep), dtype=int)

    return peaks, {}