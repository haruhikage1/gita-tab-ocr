from pydantic import BaseModel


class RecognitionResponse(BaseModel):
    success: bool
    format: str
    processing_time: float
    message: str = ""


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str