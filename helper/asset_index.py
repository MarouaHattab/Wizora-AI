import hashlib
import json
import logging
import math
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from llm.EmbeddingProviderFactory import EmbeddingProviderFactory, resolve_embedding_config
from .asset_paths import validate_asset_id
from .ingestion_errors import (
    ChunkingError,
    EmbeddingError,
    EmptyContentError,
    IndexingError,
    ProcessingError,
)
from .text_splitter import DEFAULT_SEPARATORS, SPLITTER_VERSION, split_documents


MAX_CACHED_INDEXES = 8
_INDEX_SCHEMA_VERSION = 2
_logger = logging.getLogger(__name__)
_cache_lock = RLock()
_client = None
_indexes = OrderedDict()
_live_indexes = weakref.WeakValueDictionary()
_build_locks = {}


@dataclass(frozen=True)
class AssetIndex:
    asset_id: str
    index_version: str
    chunk_count: int
    vectorstore: Chroma
    _expected_chunks: dict = field(repr=False, compare=False)
    _collection_metadata: dict = field(repr=False, compare=False)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def retrieve(self, question: str) -> list[Document]:
        with self._lock:
            if not _is_complete(self):
                raise IndexingError("The asset index is incomplete; rebuild it before retrieval")
            try:
                documents = self.vectorstore.max_marginal_relevance_search(
                    question,
                    k=min(5, self.chunk_count),
                    fetch_k=min(20, self.chunk_count),
                    filter={"$and": [
                        {"asset_id": self.asset_id},
                        {"index_version": self.index_version},
                    ]},
                )
            except ProcessingError:
                raise
            except Exception as exc:
                raise IndexingError("Unable to retrieve the asset's indexed content") from exc

            for document in documents:
                chunk_id = document.metadata.get("chunk_id")
                expected = self._expected_chunks.get(chunk_id)
                if (
                    document.metadata.get("asset_id") != self.asset_id
                    or document.metadata.get("index_version") != self.index_version
                    or expected is None
                    or document.page_content != expected[0]
                    or document.metadata != expected[1]
                    or document.id not in (None, chunk_id)
                ):
                    raise IndexingError("Retrieved content does not match the requested asset index")
                document.id = chunk_id
            return documents


def _setting(config, name):
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _source_documents(asset_id, context, documents):
    if not isinstance(context, str):
        raise EmptyContentError("The asset has no text to index")
    source_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    if documents is None:
        documents = [Document(page_content=context)]
    records = []
    for record_index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise IndexingError("Source records must be LangChain documents")
        if not document.page_content.strip():
            continue
        metadata = dict(document.metadata)
        if metadata.get("asset_id", asset_id) != asset_id:
            raise IndexingError("Source records belong to a different asset")
        metadata.update(asset_id=asset_id)
        metadata.setdefault("asset_version", source_hash)
        metadata.setdefault("source_type", "text")
        metadata.setdefault("source", asset_id)
        metadata.setdefault("record_index", record_index)
        for name, value in metadata.items():
            if (
                not isinstance(name, str)
                or not isinstance(value, (str, int, float, bool))
                or isinstance(value, float) and not math.isfinite(value)
            ):
                raise IndexingError("Source metadata must contain finite scalar values")
        metadata.setdefault("record_id", _digest({
            "text": document.page_content, "metadata": metadata,
        }))
        records.append(Document(
            page_content=document.page_content,
            metadata=metadata,
            id=document.id or metadata["record_id"],
        ))
    if not records:
        raise EmptyContentError("The asset has no text to index")
    return records, source_hash, _digest([
        {"id": record.id, "text": record.page_content, "metadata": record.metadata}
        for record in records
    ])


@contextmanager
def _building_asset(asset_id):
    with _cache_lock:
        entry = _build_locks.setdefault(asset_id, [RLock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _cache_lock:
            entry[1] -= 1
            if entry[1] == 0:
                del _build_locks[asset_id]


def _chroma_client():
    global _client
    with _cache_lock:
        if _client is None:
            _client = chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False))
        return _client


def _delete_collection(client, collection_name, lock):
    with lock:
        try:
            client.delete_collection(collection_name)
        except chromadb.errors.NotFoundError:
            pass
        except Exception:
            _logger.warning("Unable to release an unused asset index", exc_info=True)


def _is_complete(index):
    with index._lock:
        try:
            collection = index.vectorstore._collection
            metadata = index.vectorstore._client.get_collection(
                collection.name, embedding_function=None,
            ).metadata
            if metadata != index._collection_metadata or collection.count() != index.chunk_count:
                return False
            rows = collection.get(include=["documents", "metadatas"])
            if (
                set(rows["ids"]) != set(index._expected_chunks)
                or len(rows["ids"]) != index.chunk_count
                or len(rows["documents"]) != index.chunk_count
                or len(rows["metadatas"]) != index.chunk_count
            ):
                return False
            return all(
                index._expected_chunks[chunk_id] == (content, metadata)
                for chunk_id, content, metadata in zip(
                    rows["ids"], rows["documents"], rows["metadatas"], strict=True,
                )
            )
        except Exception:
            return False


def _remember(index):
    with _cache_lock:
        _indexes[index.index_version] = index
        _indexes.move_to_end(index.index_version)
        _live_indexes[index.index_version] = index
        while len(_indexes) > MAX_CACHED_INDEXES:
            _indexes.popitem(last=False)


def invalidate_asset_index(asset_id: str) -> None:
    validate_asset_id(asset_id)
    with _building_asset(asset_id), _cache_lock:
        for version, index in list(_live_indexes.items()):
            if index.asset_id == asset_id:
                _indexes.pop(version, None)
                _live_indexes.pop(version, None)


def ensure_asset_index(
    asset_id: str,
    context: str,
    config,
    *,
    documents: list[Document] | None = None,
) -> AssetIndex:
    validate_asset_id(asset_id)
    records, source_hash, records_hash = _source_documents(asset_id, context, documents)
    chunk_size = _setting(config, "CHUNK_SIZE")
    chunk_overlap = _setting(config, "CHUNK_OVERLAP")
    strategy = _setting(config, "CHUNKING_STRATEGY")
    if strategy is None:
        strategy = "recursive"
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ChunkingError("CHUNK_SIZE must be a positive integer")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ChunkingError("CHUNK_OVERLAP must be an integer from zero to CHUNK_SIZE - 1")
    if not isinstance(strategy, str) or strategy not in {"recursive", "token"}:
        raise ChunkingError("CHUNKING_STRATEGY must be recursive or token")

    with _building_asset(asset_id):
        try:
            provider, model_id = resolve_embedding_config(config)
            embedding = EmbeddingProviderFactory(config).create(provider, model_id)
            embedding_config = embedding.fingerprint
            if not isinstance(embedding_config, Mapping):
                raise EmbeddingError("The embedding resource has no configuration fingerprint")
            embedding_version = _digest(embedding_config)
        except ProcessingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Unable to initialize the configured embedding resource") from exc

        index_config = {
            "scheme": _INDEX_SCHEMA_VERSION,
            "asset_id": asset_id,
            "source_hash": source_hash,
            "records_hash": records_hash,
            "embedding": embedding_config,
            "splitter": {
                "version": SPLITTER_VERSION,
                "type": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": DEFAULT_SEPARATORS,
            },
        }
        index_version = _digest(index_config)
        with _cache_lock:
            previous = _live_indexes.get(index_version)
        if previous is not None and _is_complete(previous):
            _remember(previous)
            return previous
        if previous is not None:
            with _cache_lock:
                _indexes.pop(index_version, None)
                _live_indexes.pop(index_version, None)

        chunks = split_documents(
            records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding=embedding,
            strategy=strategy,
        )
        if not chunks:
            raise EmptyContentError("The asset produced no indexable chunks")

        ids = []
        metadatas = []
        texts = []
        expected_chunks = {}
        for chunk_index, chunk in enumerate(chunks):
            text = chunk.page_content
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{index_version}:{chunk_index}:{content_hash}".encode("utf-8")
            ).hexdigest()
            metadata = dict(chunk.metadata)
            metadata.update({
                "asset_id": asset_id,
                "index_version": index_version,
                "index_schema_version": _INDEX_SCHEMA_VERSION,
                "embedding_config_version": embedding_version,
                "record_chunk_index": metadata.get("chunk_index", 0),
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "source_hash": source_hash,
                "records_hash": records_hash,
            })
            ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)
            expected_chunks[chunk_id] = (text, metadata)

        collection_metadata = {
            "asset_id": asset_id,
            "index_version": index_version,
            "index_schema_version": _INDEX_SCHEMA_VERSION,
            "embedding_config_version": embedding_version,
            "source_hash": source_hash,
            "records_hash": records_hash,
            "chunk_count": len(ids),
            "chunk_ids_hash": _digest(ids),
        }
        try:
            client = _chroma_client()
        except Exception as exc:
            raise IndexingError("Unable to initialize the asset index storage") from exc
        collection_name = f"wizora_{index_version[:40]}_{uuid4().hex[:12]}"
        collection_lock = RLock()
        try:
            vectorstore = Chroma.from_texts(
                texts,
                embedding,
                ids=ids,
                metadatas=metadatas,
                collection_name=collection_name,
                collection_metadata=collection_metadata,
                client=client,
            )
            index = AssetIndex(
                asset_id, index_version, len(texts), vectorstore,
                expected_chunks, collection_metadata, collection_lock,
            )
            if not _is_complete(index):
                raise IndexingError("The asset index is incomplete after construction")
        except Exception as exc:
            _delete_collection(client, collection_name, collection_lock)
            if isinstance(exc, ProcessingError):
                raise
            raise IndexingError("Unable to construct the asset index") from exc

        weakref.finalize(index, _delete_collection, client, collection_name, collection_lock)
        _remember(index)
        return index
