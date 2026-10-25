from threading import Lock
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from helper.failures import StorageUnavailable
from helper.session_store import LeaseConflict, acquire_lease, release_lease

_registry_lock = Lock()
_session_locks = {}
_consumed_reviews = {}


class SessionLease:
    def __init__(self, lock, workflow, thread_id, owner):
        self.lock = lock
        self.workflow = workflow
        self.thread_id = thread_id
        self.owner = owner
        self.released = False

    def release(self):
        with _registry_lock:
            if self.released:
                return
            self.released = True
        try:
            release_lease(self.workflow, self.thread_id, self.owner)
        finally:
            self.lock.release()


class SessionStreamingResponse(StreamingResponse):
    def __init__(self, content, *, session_lock, **kwargs):
        super().__init__(content, **kwargs)
        self.session_lock = session_lock

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    await close()
            finally:
                self.session_lock.release()


def acquire_session(workflow, thread_id):
    key = (workflow, thread_id)
    with _registry_lock:
        lock = _session_locks.setdefault(key, Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Session is already processing a request")
    owner = str(uuid4())
    try:
        acquire_lease(workflow, thread_id, owner)
    except LeaseConflict:
        lock.release()
        raise HTTPException(
            status_code=409, detail="Session is already processing a request"
        ) from None
    except StorageUnavailable:
        lock.release()
        raise HTTPException(status_code=503, detail="Workflow storage is unavailable") from None
    except Exception:
        lock.release()
        raise
    return SessionLease(lock, workflow, thread_id, owner)


def session_status(snapshot):
    if not snapshot or not snapshot.values:
        return "not_found"
    if any(task.error for task in snapshot.tasks):
        return "failed"
    if any(task.interrupts for task in snapshot.tasks):
        return "interrupted_for_feedback"
    if not snapshot.next:
        return "completed"
    return "running"


def review_revision(snapshot):
    if session_status(snapshot) != "interrupted_for_feedback":
        return None
    return snapshot.config["configurable"]["checkpoint_id"]


def validate_feedback(workflow, thread_id, snapshot, feedback, review_id):
    status = session_status(snapshot)
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Session not found")
    if status == "completed":
        raise HTTPException(status_code=409, detail="Session is already completed")
    if status != "interrupted_for_feedback":
        raise HTTPException(status_code=409, detail="Session has no pending review")
    revision = review_revision(snapshot)
    if review_id is not None and review_id != revision:
        raise HTTPException(status_code=409, detail="Feedback targets a stale review")
    if not feedback.strip():
        raise HTTPException(status_code=422, detail="Feedback must not be empty")
    key = (workflow, thread_id)
    consumed = _consumed_reviews.get(key, set())
    if revision in consumed:
        raise HTTPException(
            status_code=409, detail="Feedback for this review has already been accepted"
        )
    if review_id is None and consumed:
        raise HTTPException(
            status_code=409,
            detail="The current review_id is required after the first review",
        )


def accept_feedback(workflow, thread_id, snapshot, feedback, review_id):
    validate_feedback(workflow, thread_id, snapshot, feedback, review_id)
    _consumed_reviews.setdefault((workflow, thread_id), set()).add(review_revision(snapshot))
