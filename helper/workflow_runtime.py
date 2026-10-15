import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from helper import get_settings
from helper.failures import StorageUnavailable
from helper.session_store import (
    close_store, delete_session_if_expired, expired_threads, is_expired, open_store, ping, record_session,
)
from helper.workflow_logging import configure_logging
from helper.workflow_metrics import metrics


logger = logging.getLogger("wizora.runtime")
_STATUS = {
    "interrupted_for_feedback": "awaiting_feedback",
    "completed": "completed",
    "failed": "failed",
    "running": "running",
}


class WorkflowRuntime:
    def __init__(self):
        self.ready = False
        self.graphs = {}
        self.saver = None
        self._context = None
        self._lock_handle = None

    async def start(self):
        if self.ready:
            return
        assert_single_worker()
        settings = get_settings()
        configure_logging(settings)
        path = resolve_checkpoint_path(settings)
        context = None
        entered = False
        try:
            self._lock_handle = acquire_process_lock(path.parent / "wizora.process.lock")
            context = AsyncSqliteSaver.from_conn_string(str(path))
            saver = await context.__aenter__()
            entered = True
            await saver.setup()
            open_store(path, settings.SESSION_LEASE_SECONDS)
            self.saver = saver
            self._context = context
            self.graphs = _compile(saver)
            try:
                removed = await self._cleanup(settings)
            except Exception:
                logger.warning("session retention could not be completed", exc_info=False)
                removed = 0
            if not ping():
                raise StorageUnavailable("Workflow storage is unavailable")
            self.ready = True
            logger.info(
                "workflow runtime ready checkpoint=sqlite deployment=single_process expired_sessions=%s",
                removed,
            )
        except Exception as exc:
            self.ready = False
            self.graphs = {}
            self.saver = None
            close_store()
            if entered and context is not None:
                await context.__aexit__(None, None, None)
            self._context = None
            release_process_lock(self._lock_handle)
            self._lock_handle = None
            logger.warning(
                "workflow runtime failed during startup exception_type=%s",
                type(exc).__name__,
            )
            raise

    async def stop(self):
        self.ready = False
        self.graphs = {}
        saver_context = self._context
        self._context = None
        self.saver = None
        close_store()
        if saver_context is not None:
            await saver_context.__aexit__(None, None, None)
        release_process_lock(self._lock_handle)
        self._lock_handle = None
        logger.info("workflow runtime stopped")

    async def healthy(self):
        if not self.ready or self.saver is None:
            return False
        try:
            await self.saver.aget_tuple({
                "configurable": {"thread_id": "__wizora_health__", "checkpoint_ns": ""},
            })
        except Exception:
            return False
        return ping()

    async def _cleanup(self, settings):
        if settings.SESSION_RETENTION_DAYS <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.SESSION_RETENTION_DAYS)).isoformat()
        removed = 0
        for workflow, thread_id in expired_threads(cutoff):
            if not is_expired(workflow, thread_id, cutoff):
                continue
            await self.saver.adelete_thread(thread_id)
            if delete_session_if_expired(workflow, thread_id, cutoff):
                removed += 1
                logger.info("expired session removed workflow=%s thread=%s", workflow, thread_id)
        return removed


def require_graph(name):
    graph = workflow_runtime.graphs.get(name)
    if graph is None or not workflow_runtime.ready:
        raise StorageUnavailable("Workflow storage is not ready")
    return graph


def note_snapshot(workflow, thread_id, snapshot, *, started=None, error=None, request_id=None):
    from helper.session_review import review_revision, session_status
    from helper.sse import public_error

    public = session_status(snapshot) if snapshot is not None else "failed"
    if public in _STATUS:
        status = _STATUS[public]
        preserve_terminal = False
    elif error is not None:
        status = "failed"
        preserve_terminal = snapshot is None
    else:
        return
    category = public_error(error)["category"] if error is not None else None
    asset_id = _asset_id(snapshot)
    review_id = review_revision(snapshot) if snapshot is not None else None
    try:
        record_session(
            workflow, thread_id, status=status, review_id=review_id, asset_id=asset_id,
            error_category=category, preserve_terminal=preserve_terminal,
        )
    except StorageUnavailable:
        logger.warning(
            "session metadata was not recorded workflow=%s thread=%s status=%s",
            workflow, thread_id, status,
        )
    elapsed = None if started is None else round((perf_counter() - started) * 1000, 3)
    if elapsed is not None:
        metrics.observe("workflow_ms", elapsed, workflow=workflow, status=status)
    if status == "awaiting_feedback":
        metrics.increment("workflow_interrupt", workflow=workflow)
    elif status == "completed":
        metrics.increment("workflow_completed", workflow=workflow)
    elif status == "failed":
        metrics.increment("workflow_failed", workflow=workflow)
    logger.info(
        "workflow snapshot request=%s workflow=%s thread=%s status=%s elapsed_ms=%s error=%s",
        request_id or "-", workflow, thread_id, status, elapsed if elapsed is not None else "-",
        category or "-",
    )


def assert_single_worker():
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        raw = os.getenv(name, "").strip()
        if raw and raw != "1":
            raise RuntimeError(f"{name}={raw} is unsupported; Wizora-AI runs as one process")


def resolve_checkpoint_path(settings):
    raw = settings.CHECKPOINT_DATABASE.strip()
    if not raw:
        raise RuntimeError("CHECKPOINT_DATABASE is required")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path = path.resolve()
    if path.exists() and path.is_dir():
        raise RuntimeError("CHECKPOINT_DATABASE must be a SQLite file path, not a directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def acquire_process_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.seek(0, os.SEEK_END) < 1:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another Wizora-AI process is already running for this checkpoint database") from exc
    return handle


def release_process_lock(handle):
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()


def _compile(saver):
    from controllers.QuestionAnswerGraphController import compile_qa_graph
    from controllers.QuestionGenGraphController import compile_question_graph
    from controllers.SummarizerGenGraphController import compile_summary_graph

    return {
        "question_generation": compile_question_graph(saver),
        "summarization": compile_summary_graph(saver),
        "qa": compile_qa_graph(saver),
    }


def _asset_id(snapshot):
    values = getattr(snapshot, "values", None)
    if not isinstance(values, dict):
        return None
    asset_id = values.get("asset_id")
    return asset_id if isinstance(asset_id, str) else None


workflow_runtime = WorkflowRuntime()
