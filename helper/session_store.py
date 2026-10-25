import sqlite3
from datetime import datetime, timezone
from threading import Lock

from helper.failures import StorageUnavailable


class LeaseConflict(RuntimeError):
    pass


_db_lock = Lock()
_conn = None
_lease_seconds = 210.0
_TERMINAL = {"awaiting_feedback", "completed", "running"}


def open_store(path, lease_seconds):
    global _conn, _lease_seconds
    if lease_seconds <= 0:
        raise StorageUnavailable("Session lease duration is invalid")
    conn = sqlite3.connect(path, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS wizora_session (
                workflow TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                status TEXT NOT NULL,
                review_id TEXT,
                asset_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                last_error TEXT,
                PRIMARY KEY (workflow, thread_id)
            );
            CREATE TABLE IF NOT EXISTS wizora_lease (
                workflow TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                acquired_at REAL NOT NULL,
                PRIMARY KEY (workflow, thread_id)
            );
            """
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.close()
        raise StorageUnavailable("Workflow storage is unavailable") from exc
    with _db_lock:
        if _conn is not None:
            conn.close()
            return
        _conn = conn
        _lease_seconds = float(lease_seconds)


def close_store():
    global _conn
    with _db_lock:
        conn = _conn
        _conn = None
    if conn is not None:
        conn.close()


def ping():
    with _db_lock:
        conn = _conn
        if conn is None:
            return False
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False


def acquire_lease(workflow, thread_id, owner):
    now = datetime.now(timezone.utc).timestamp()
    with _db_lock:
        conn = _connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner, acquired_at FROM wizora_lease WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            ).fetchone()
            if (
                row is not None
                and row["owner"] != owner
                and now - float(row["acquired_at"]) < _lease_seconds
            ):
                conn.rollback()
                raise LeaseConflict("Session is already processing a request")
            conn.execute(
                "DELETE FROM wizora_lease WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            )
            conn.execute(
                "INSERT INTO wizora_lease (workflow, thread_id, owner, acquired_at) VALUES (?, ?, ?, ?)",
                (workflow, thread_id, owner, now),
            )
            conn.commit()
        except LeaseConflict:
            raise
        except sqlite3.Error as exc:
            _rollback(conn)
            raise StorageUnavailable("Workflow storage is unavailable") from exc


def release_lease(workflow, thread_id, owner):
    with _db_lock:
        conn = _conn
        if conn is None:
            return
        try:
            conn.execute(
                "DELETE FROM wizora_lease WHERE workflow=? AND thread_id=? AND owner=?",
                (workflow, thread_id, owner),
            )
            conn.commit()
        except sqlite3.Error:
            _rollback(conn)


def record_session(
    workflow,
    thread_id,
    *,
    status,
    review_id=None,
    asset_id=None,
    error_category=None,
    preserve_terminal=False,
):
    now = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        conn = _conn
        if conn is None:
            raise StorageUnavailable("Workflow storage is not ready")
        try:
            existing = conn.execute(
                "SELECT * FROM wizora_session WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            ).fetchone()
            if (
                preserve_terminal
                and existing is not None
                and status == "failed"
                and existing["status"] in _TERMINAL
            ):
                status = existing["status"]
            created = existing["created_at"] if existing is not None else now
            completed = existing["completed_at"] if existing is not None else None
            if status == "completed" and completed is None:
                completed = now
            if status == "completed":
                review_id = None
            elif review_id is None and existing is not None:
                review_id = existing["review_id"]
            if asset_id is None and existing is not None:
                asset_id = existing["asset_id"]
            conn.execute(
                """
                INSERT INTO wizora_session (
                    workflow, thread_id, status, review_id, asset_id,
                    created_at, updated_at, completed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow, thread_id) DO UPDATE SET
                    status=excluded.status,
                    review_id=excluded.review_id,
                    asset_id=excluded.asset_id,
                    updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at,
                    last_error=excluded.last_error
                """,
                (
                    workflow,
                    thread_id,
                    status,
                    review_id,
                    asset_id,
                    created,
                    now,
                    completed,
                    error_category,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            _rollback(conn)
            raise StorageUnavailable("Workflow storage is unavailable") from exc


def fetch_session(workflow, thread_id):
    with _db_lock:
        conn = _conn
        if conn is None:
            return None
        row = conn.execute(
            "SELECT * FROM wizora_session WHERE workflow=? AND thread_id=?",
            (workflow, thread_id),
        ).fetchone()
    return dict(row) if row is not None else None


def completed_sessions(workflow, asset_id=None):
    with _db_lock:
        conn = _connection()
        try:
            rows = conn.execute(
                """
                SELECT thread_id, asset_id, completed_at FROM wizora_session
                WHERE workflow=? AND status='completed' AND (? IS NULL OR asset_id=?)
                ORDER BY completed_at DESC, thread_id
                """,
                (workflow, asset_id, asset_id),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageUnavailable("Workflow storage is unavailable") from exc
    return [dict(row) for row in rows]


def is_expired(workflow, thread_id, cutoff):
    with _db_lock:
        conn = _conn
        if conn is None:
            return False
        row = conn.execute(
            "SELECT status, updated_at FROM wizora_session WHERE workflow=? AND thread_id=?",
            (workflow, thread_id),
        ).fetchone()
    return (
        row is not None and row["status"] in {"completed", "failed"} and row["updated_at"] < cutoff
    )


def expired_threads(cutoff):
    with _db_lock:
        conn = _connection()
        rows = conn.execute(
            """
            SELECT workflow, thread_id FROM wizora_session
            WHERE status IN ('completed', 'failed') AND updated_at < ?
            """,
            (cutoff,),
        ).fetchall()
    return [(row["workflow"], row["thread_id"]) for row in rows]


def delete_session_if_expired(workflow, thread_id, cutoff):
    with _db_lock:
        conn = _connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, updated_at FROM wizora_session WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            ).fetchone()
            if (
                row is None
                or row["status"] not in {"completed", "failed"}
                or row["updated_at"] >= cutoff
            ):
                conn.rollback()
                return False
            conn.execute(
                "DELETE FROM wizora_session WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            )
            conn.execute(
                "DELETE FROM wizora_lease WHERE workflow=? AND thread_id=?",
                (workflow, thread_id),
            )
            conn.commit()
            return True
        except sqlite3.Error as exc:
            _rollback(conn)
            raise StorageUnavailable("Workflow storage is unavailable") from exc


def _connection():
    if _conn is None:
        raise StorageUnavailable("Workflow storage is not ready")
    return _conn


def _rollback(conn):
    try:
        conn.rollback()
    except sqlite3.Error:
        return
