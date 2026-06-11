from fastapi import FastAPI
from gtrs.api.routes import router

app = FastAPI(
    title="GTRS API",
    description="Guitar Tablature Recognition System API",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}