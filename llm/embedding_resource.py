import json
from math import isfinite
from numbers import Integral
from threading import RLock

from langchain_core.embeddings import Embeddings

from helper.ingestion_errors import EmbeddingError, ModelInitializationError


class EmbeddingResource(Embeddings):
    def __init__(
        self,
        model,
        *,
        provider: str,
        model_id: str,
        pipeline: str,
        normalize_embeddings: bool,
        delegate: Embeddings | None = None,
        expected_dimension: int | None = None,
    ):
        self.embedding_model = model
        self.model_id = model_id
        self._delegate = delegate
        self._lock = RLock()
        self._replace_newlines = delegate is not None
        try:
            self.tokenizer = model.tokenizer
            limit = model.max_seq_length
            dimension = model.get_sentence_embedding_dimension()
            if isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0:
                raise ValueError("Invalid model token limit")
            if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0:
                raise ValueError("Invalid embedding dimension")
            if expected_dimension is not None and (
                isinstance(expected_dimension, bool)
                or not isinstance(expected_dimension, Integral)
                or expected_dimension != dimension
            ):
                raise ValueError("Configured embedding dimension differs from the model")
            self.max_tokens = int(limit)
            self.embedding_size = int(dimension)
            first_module = model._first_module() if hasattr(model, "_first_module") else None
            self._lowercase = bool(getattr(first_module, "do_lower_case", False))
            default_prompt_name = getattr(model, "default_prompt_name", None)
            self._prompt = (
                model.prompts[default_prompt_name] if default_prompt_name is not None else ""
            )
            if not isinstance(self._prompt, str):
                raise ValueError("Invalid model prompt")
            modules = []
            for module in getattr(model, "_modules", {}).values():
                description = {"type": type(module).__name__}
                get_config = getattr(module, "get_config_dict", None)
                if callable(get_config):
                    module_config = get_config()
                    json.dumps(module_config, sort_keys=True)
                    description["config"] = module_config
                modules.append(description)
            self._fingerprint = {
                "provider": provider,
                "model_id": model_id,
                "pipeline": pipeline,
                "max_seq_length": self.max_tokens,
                "dimension": self.embedding_size,
                "normalize_embeddings": normalize_embeddings,
                "default_prompt": self._prompt,
                "lowercase": self._lowercase,
                "replace_newlines": self._replace_newlines,
                "modules": modules,
            }
            revision = getattr(getattr(first_module, "auto_model", None), "config", None)
            revision = getattr(revision, "_commit_hash", None)
            if isinstance(revision, str) and revision:
                self._fingerprint["revision"] = revision
            self.token_count("")
        except Exception as exc:
            raise ModelInitializationError("The embedding model configuration could not be initialized.") from exc

    @property
    def fingerprint(self) -> dict:
        return json.loads(json.dumps(self._fingerprint))

    @property
    def max_seq_length(self) -> int:
        return self.max_tokens

    def token_count(self, text: str) -> int:
        if not isinstance(text, str):
            raise EmbeddingError("Embedding input must be text.")
        with self._lock:
            try:
                if self._replace_newlines:
                    text = text.replace("\n", " ")
                text = (self._prompt + text).strip()
                if self._lowercase:
                    text = text.lower()
                return len(self.tokenizer.encode(
                    text, add_special_tokens=True, truncation=False, verbose=False
                ))
            except Exception as exc:
                raise EmbeddingError("The embedding input could not be tokenized.") from exc

    def _validate_inputs(self, texts: list[str]) -> None:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError("Embedding input must contain non-empty text.")
            if self.token_count(text) > self.max_tokens:
                raise EmbeddingError(
                    f"Embedding input exceeds the model limit of {self.max_tokens} tokens."
                )

    def _validate_vectors(self, vectors, expected_count: int) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if len(vectors) != expected_count:
            raise EmbeddingError("The embedding model returned an unexpected number of vectors.")
        result = []
        for vector in vectors:
            if len(vector) != self.embedding_size:
                raise EmbeddingError("The embedding model returned an inconsistent vector dimension.")
            values = [float(value) for value in vector]
            if not all(isfinite(value) for value in values):
                raise EmbeddingError("The embedding model returned non-finite vector values.")
            result.append(values)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            self._validate_inputs(texts)
            try:
                vectors = (
                    self._delegate.embed_documents(texts)
                    if self._delegate is not None
                    else self.embedding_model.encode(texts, convert_to_numpy=True)
                )
                return self._validate_vectors(vectors, len(texts))
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("The document embeddings could not be generated.") from exc

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            self._validate_inputs([text])
            try:
                vectors = (
                    [self._delegate.embed_query(text)]
                    if self._delegate is not None
                    else self.embedding_model.encode([text], convert_to_numpy=True)
                )
                return self._validate_vectors(vectors, 1)[0]
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError("The query embedding could not be generated.") from exc
