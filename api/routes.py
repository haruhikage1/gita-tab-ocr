import os
import tempfile
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from gtrs.api.schemas import (
    HealthResponse,
    RecognitionRequest,
    RecognitionResponse,
    TaskStatusResponse,
)
from gtrs.simple_logging import eprint

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

_tasks: dict[str, dict[str, Any]] = {}


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize(
    file: UploadFile = File(...),
    format: str = "musicxml",
    tuning: str = "standard",
    async_mode: bool = False,
) -> Any:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)} bytes. Max: {MAX_FILE_SIZE} bytes",
        )

    if async_mode:
        task_id = str(uuid.uuid4())
        _tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "result": None,
        }
        return TaskStatusResponse(
            task_id=task_id,
            status="processing",
            message="Task submitted for async processing",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        from gtrs.main import OutputFormat, ProcessingConfig, process_image

        output_format = OutputFormat(format)
        config = ProcessingConfig(
            enable_debug=False,
            enable_cache=False,
            use_gpu_inference=False,
            output_format=output_format,
            tuning_name=tuning,
            output_dir=os.path.dirname(tmp_path),
        )

        t0 = time.time()
        process_image(tmp_path, config)
        elapsed = time.time() - t0

        return RecognitionResponse(
            success=True,
            format=format,
            processing_time=round(elapsed, 2),
            message="Recognition complete",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str) -> Any:
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    elapsed = time.time() - task["created_at"]

    if elapsed > 60:
        task["status"] = "failed"
        task["error"] = "Timeout"

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Any:
    return HealthResponse(status="healthy", version="0.1.0")