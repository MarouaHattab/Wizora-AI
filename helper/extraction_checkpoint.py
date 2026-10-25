import hashlib
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document


class ChunkCheckpoint:
    def __init__(self, source, namespace, config):
        source = Path(source)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode())
        self.directory = source.parent / ".extraction-checkpoints" / namespace / digest.hexdigest()

    def load(self, key):
        path = self.directory / (str(int(key)) + ".json")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), list):
                return None
            return [Document(**item) for item in payload["records"]]
        except (OSError, ValueError, TypeError):
            return None

    def save(self, key, records):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (str(int(key)) + ".json")
        temporary = path.with_suffix("." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {"schema_version": 1, "records": [record.model_dump() for record in records]},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
