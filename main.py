import sqlite3
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from helper.failures import StorageUnavailable
from routes import (
    batch_processing_router,
    file_processing_router,
    job_results_router,
    provider_status_router,
    qa_router,
    question_gen_router,
    session_router,
    summarizer_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from helper.workflow_runtime import workflow_runtime

    await workflow_runtime.start()
    try:
        import anyio

        from helper.asset_catalog import recover_interrupted_uploads
        from helper.job_results import recover_interrupted_jobs

        await anyio.to_thread.run_sync(recover_interrupted_uploads)
        await anyio.to_thread.run_sync(recover_interrupted_jobs)
        yield
    finally:
        await workflow_runtime.stop()


app = FastAPI(
    title="Wizora-AI API",
    version="1.0.0",
    description="API for Wizora-AI agentic system including preprocessing and graph-based agents.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(provider_status_router, prefix="/providers", tags=["AI Providers"])
api_v1_router.include_router(session_router, prefix="/sessions", tags=["Sessions"])
api_v1_router.include_router(job_results_router, prefix="/preprocessing", tags=["Job Results"])
api_v1_router.include_router(
    batch_processing_router, prefix="/preprocessing", tags=["Batch Processing"]
)

app.include_router(file_processing_router, tags=["File Processing"])

api_v1_router.include_router(
    file_processing_router, prefix="/preprocessing", tags=["Preprocessing"]
)

graph_base_router = APIRouter(prefix="/graph", tags=["Graph Agents"])

graph_base_router.include_router(
    question_gen_router, prefix="/qg", tags=["Question Generation Agent"]
)
graph_base_router.include_router(
    summarizer_router, prefix="/summarizer", tags=["Summarization Agent"]
)
graph_base_router.include_router(qa_router, prefix="/qa", tags=["Question Answering Agent"])

api_v1_router.include_router(graph_base_router)
app.include_router(api_v1_router)


@app.exception_handler(StorageUnavailable)
async def storage_unavailable_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Workflow storage is unavailable"})


@app.exception_handler(sqlite3.OperationalError)
async def sqlite_unavailable_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": "Workflow storage is unavailable"})


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Wizora-AI API."}


@app.get("/health/live", tags=["Health"])
async def health_live():
    return {"status": "live"}


@app.get("/health/ready", tags=["Health"])
async def health_ready():
    from helper.workflow_runtime import workflow_runtime

    if not await workflow_runtime.healthy():
        raise HTTPException(status_code=503, detail="Workflow storage is unavailable")
    return {"status": "ready", "checkpoint": "sqlite", "deployment": "single_process"}


@app.get("/health/metrics", tags=["Health"])
async def health_metrics():
    from helper.workflow_metrics import metrics
    from helper.workflow_runtime import workflow_runtime

    if not workflow_runtime.ready:
        raise HTTPException(status_code=503, detail="Workflow storage is unavailable")
    return metrics.snapshot()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1)
