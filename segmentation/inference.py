import hashlib
import lzma
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from gtrs.model import TabSegmentationResult
from gtrs.segmentation.categories import NUM_TAB_CATEGORIES, TabCategory
from gtrs.segmentation import segnet_path_onnx, segnet_path_onnx_fp16, segmentation_version
from gtrs.simple_logging import eprint
from gtrs.type_definitions import NDArray


class TabSegnet:
    def __init__(self, use_gpu_inference: bool) -> None:
        self.use_gpu = False
        if use_gpu_inference:
            try:
                ort.preload_dlls()
                self.model = ort.InferenceSession(
                    segnet_path_onnx_fp16, providers=["CUDAExecutionProvider"]
                )
                self.fp16 = True
                self.use_gpu = True
            except Exception as e:
                eprint(f"Error loading CUDA model: {e}")
                self.model = ort.InferenceSession(segnet_path_onnx_fp16)
                self.fp16 = True
        else:
            self.model = ort.InferenceSession(segnet_path_onnx)
            self.fp16 = False

        self.io_binding = self.model.io_binding()
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def run(self, input_data: NDArray) -> NDArray:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", input_data.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", input_data.astype(np.float32))
        self.io_binding.bind_output("output", "cpu")
        self.model.run_with_iobinding(self.io_binding)
        out = self.io_binding.get_outputs()[0].numpy()
        return out


def extract_patch(image: NDArray, y: int, x: int, win_size: int) -> NDArray:
    c, h, w = image.shape
    patch = np.full((c, win_size, win_size), 255, dtype=image.dtype)
    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + win_size, h)
    x1 = min(x + win_size, w)
    py0 = 0
    px0 = 0
    py1 = py0 + (y1 - y0)
    px1 = px0 + (x1 - x0)
    patch[:, py0:py1, px0:px1] = image[:, y0:y1, x0:x1]
    return patch


def merge_patches(
    patches: list[NDArray], image_shape: tuple[int, int], win_size: int, step_size: int
) -> NDArray:
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        y = min(iy, image_shape[0] - win_size)
        y0 = max(y, 0)
        y1 = min(y + win_size, image_shape[0])

        for ix in range(0, image_shape[1], step_size):
            x = min(ix, image_shape[1] - win_size)
            x0 = max(x, 0)
            x1 = min(x + win_size, image_shape[1])

            patch = patches[idx]
            ph = y1 - y0
            pw = x1 - x0

            reconstructed[y0:y1, x0:x1] += patch[:ph, :pw]
            weight[y0:y1, x0:x1] += 1
            idx += 1

    weight[weight == 0] = 1
    reconstructed /= weight
    return reconstructed.astype(patches[0].dtype)


def inference(
    image_org: NDArray,
    use_gpu_inference: bool,
    batch_size: int = 8,
    step_size: int = 240,
    win_size: int = 320,
) -> NDArray:
    eprint("Starting Tab Segnet Inference.")
    t0 = perf_counter()

    model = TabSegnet(use_gpu_inference)

    image_org = cv2.cvtColor(image_org, cv2.COLOR_GRAY2BGR)
    image = np.transpose(image_org, (2, 0, 1)).astype(np.float32)

    c, h, w = image.shape
    data: list[NDArray] = []
    batch: list[NDArray] = []

    for y_loop in range(0, max(h, win_size), step_size):
        y = min(y_loop, h - win_size)
        for x_loop in range(0, max(w, win_size), step_size):
            x = min(x_loop, w - win_size)
            hop = extract_patch(image, y, x, win_size)
            batch.append(hop)

            if len(batch) == batch_size:
                batch_out = model.run(np.stack(batch, axis=0))
                for out in batch_out:
                    data.append(np.argmax(out, axis=0))
                batch.clear()

    if batch:
        batch_out = model.run(np.stack(batch, axis=0))
        for out in batch_out:
            data.append(np.argmax(out, axis=0))

    eprint(f"Tab Segnet Inference time: {perf_counter() - t0:.2f}s")

    merged = merge_patches(
        data, (int(image_org.shape[0]), int(image_org.shape[1])), win_size, step_size
    )

    return merged.astype(np.uint8)


def extract(
    original_image: NDArray,
    img_path_str: str,
    use_cache: bool = False,
    use_gpu_inference: bool = True,
    batch_size: int = 8,
    step_size: int = 240,
    win_size: int = 320,
) -> TabSegmentationResult:
    img_path = Path(img_path_str)
    f_name = os.path.splitext(img_path.name)[0]
    npy_path = img_path.parent / f"{f_name}_tab.npy"
    loaded_from_cache = False

    if npy_path.exists() and use_cache:
        eprint("Found a cache")
        file_hash = hashlib.sha256(original_image).hexdigest()
        with lzma.open(npy_path, "rb") as f:
            mask = np.load(f)
            cached_file_hash = f.readline().decode().strip()
            model_name = f.readline().decode().strip()
            if cached_file_hash == "" or model_name == "":
                eprint("Cache is missing meta information, skipping cache")
            elif file_hash != cached_file_hash:
                eprint("File hash mismatch, skipping cache")
            elif model_name != segmentation_version:
                eprint("Models have been updated, skipping cache")
            else:
                loaded_from_cache = True
                eprint("Loading from cache")

    if not loaded_from_cache:
        mask = inference(
            original_image,
            use_gpu_inference=use_gpu_inference,
            batch_size=batch_size,
            step_size=step_size,
            win_size=win_size,
        )
        if use_cache:
            eprint("Saving cache")
            file_hash = hashlib.sha256(original_image).hexdigest()
            with lzma.open(npy_path, "wb") as f:
                np.save(f, mask)
                f.write((file_hash + "\n").encode())
                f.write((segmentation_version + "\n").encode())

    background_ratio = np.sum(mask == TabCategory.BACKGROUND) / mask.size
    if background_ratio > 0.99:
        raise RuntimeError("NO_TAB_CONTENT: Background占比>99%, 未检测到Tab谱内容")

    return TabSegmentationResult(mask=mask)