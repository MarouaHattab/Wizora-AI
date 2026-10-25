import hashlib
import secrets
import time
import zipfile
from pathlib import Path
from threading import RLock

import gradio as gr
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.routing import APIRoute


class ExportRegistry:
    def __init__(self, ttl=600, maximum=16, max_bytes=64 * 1024 * 1024):
        self.ttl, self.maximum, self.max_bytes = ttl, maximum, max_bytes
        self._items, self._lock = {}, RLock()

    def register(self, path):
        payload = Path(path).read_bytes()
        if len(payload) > self.max_bytes:
            raise ValueError(
                "ZIP export exceeds the browser transport budget; download selected JSON files instead."
            )
        import io

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None or not archive.namelist():
                raise ValueError("ZIP export did not pass integrity validation.")
        with self._lock:
            self._prune()
            while (
                len(self._items) >= self.maximum
                or sum(len(item[1]) for item in self._items.values()) + len(payload)
                > self.max_bytes
            ):
                self._items.pop(next(iter(self._items)))
            token = secrets.token_urlsafe(32)
            self._items[token] = (time.monotonic() + self.ttl, payload)
        return {
            "path": "gradio_api/wizora-exports/" + token,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "filename": "wizora-job-results.zip",
        }

    def _prune(self):
        now = time.monotonic()
        for token, (expires, _) in list(self._items.items()):
            if expires <= now:
                del self._items[token]

    def read(self, token):
        with self._lock:
            self._prune()
            entry = self._items.get(token)
            if entry is None:
                raise HTTPException(
                    status_code=404,
                    detail="Export unavailable or expired. Export the results again.",
                )
            return entry[1]


EXPORTS = ExportRegistry()


async def registered_export(token: str):
    return Response(
        EXPORTS.read(token),
        media_type="application/zip",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


class ExportBlocks(gr.Blocks):
    def launch(self, *args, **kwargs):
        app_kwargs = dict(kwargs.pop("app_kwargs", None) or {})
        app_kwargs["routes"] = [
            *app_kwargs.get("routes", []),
            APIRoute(
                "/gradio_api/wizora-exports/{token}",
                registered_export,
                methods=["GET"],
                include_in_schema=False,
            ),
        ]
        return super().launch(*args, app_kwargs=app_kwargs, **kwargs)


SAVE_ZIP_JS = """async (entry) => {
  if (!entry || !entry.path) return 'Export the results first.';
  try {
    const base = window.location.origin + window.location.pathname.replace(/\/?$/, '/');
    const url = new URL(entry.path, base);
    if (url.origin !== window.location.origin) throw new Error('Invalid export origin');
    const response = await fetch(url, {credentials: 'same-origin', cache: 'no-store'});
    if (!response.ok) throw new Error('Export expired or unavailable. Export the results again.');
    const data = await response.arrayBuffer();
    const bytes = new Uint8Array(data);
    if (data.byteLength !== entry.bytes || bytes[0] !== 80 || bytes[1] !== 75)
      throw new Error('Download was intercepted or incomplete. No file was saved.');
    if (globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest('SHA-256', data);
      const actual = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
      if (actual !== entry.sha256) throw new Error('ZIP integrity check failed.');
    }
    const blob = URL.createObjectURL(new Blob([data], {type:'application/zip'}));
    const anchor = document.createElement('a');
    anchor.href = blob; anchor.download = entry.filename;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(blob), 60000);
    return '✅ ZIP verified and sent to your browser downloads.';
  } catch (error) { return '❌ ' + error.message; }
}"""
