import logging
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, StringConstraints, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Temperature = Annotated[float, Field(ge=0, allow_inf_nan=False)]
LLMProvider = Literal["OPENAI_COMPATIBLE", "NVIDIA_NIM", "OLLAMA", "GOOGLE_GENAI", "GROQ"]
EmbeddingProvider = Literal["LOCAL_EMBEDDING", "HUGGINGFACE"]


class Settings(BaseSettings):
    # API keys and tokens
    LLM_PROVIDER: LLMProvider = "OPENAI_COMPATIBLE"
    OPENAI_COMPATIBLE_API_KEY: SecretStr | None = None
    OPENAI_COMPATIBLE_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_COMPATIBLE_MODEL_ID: ModelName = "gpt-4o-mini"
    OPENAI_COMPATIBLE_TEMPERATURE: Temperature = 0.2
    NVIDIA_NIM_API_KEY: SecretStr | None = None
    NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    GROQ_API_KEY: SecretStr | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_GENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OLLAMA_API_KEY: SecretStr | None = None
    HUGGING_FACE_HUB_TOKEN: SecretStr | None = None

    # LangSmith configuration
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str | None = None
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str | None = None

    # Question Generation Agent configuration
    QUESTION_GENERATER_PROVIDER: LLMProvider
    QUESTION_GENERATER_MODEL_ID: ModelName
    QUESTION_GENERATER_TEMPERATURE: Temperature

    QUESTION_REFINER_PROVIDER: LLMProvider
    QUESTION_REFINER_MODEL_ID: ModelName
    QUESTION_REFINER_TEMPERATURE: Temperature

    QUESTION_REWRITER_PROVIDER: LLMProvider
    QUESTION_REWRITER_MODEL_ID: ModelName
    QUESTION_REWRITER_TEMPERATURE: Temperature

    # Summarizer Agent configuration
    SUMMARIZER_MAINPOINT_PROVIDER: LLMProvider
    SUMMARIZER_MAINPOINT_MODEL_ID: ModelName
    SUMMARIZER_MAINPOINT_TEMPERATURE: Temperature

    SUMMARIZER_WRITER_PROVIDER: LLMProvider
    SUMMARIZER_WRITER_MODEL_ID: ModelName
    SUMMARIZER_WRITER_TEMPERATURE: Temperature

    SUMMARIZER_REWRITER_PROVIDER: LLMProvider
    SUMMARIZER_REWRITER_MODEL_ID: ModelName
    SUMMARIZER_REWRITER_TEMPERATURE: Temperature

    # Question Answering Agent configuration
    GRADE_QA_PROVIDER: LLMProvider
    GRADE_QA_MODEL_ID: ModelName
    GRADE_QA_TEMPERATURE: Temperature

    QUESTION_ANSWERER_PROVIDER: LLMProvider
    QUESTION_ANSWERER_MODEL_ID: ModelName
    QUESTION_ANSWERER_TEMPERATURE: Temperature

    EMBEDDING_MODEL_ID: ModelName
    EMBEDDING_MODEL_PROVIDER: EmbeddingProvider
    CHUNK_SIZE: int = Field(gt=0)
    CHUNK_OVERLAP: int = Field(ge=0)
    CHUNKING_STRATEGY: Literal["recursive", "token"] = "recursive"

    LLM_TIMEOUT_SECONDS: float = Field(default=60, gt=0, allow_inf_nan=False)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=3)
    LLM_MAX_CONCURRENCY: int = Field(default=2, ge=1)
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=2048, ge=1)
    LLM_RETRY_MAX_WAIT_SECONDS: float = Field(default=10, gt=0, allow_inf_nan=False)
    # Metadata checks run at request dispatch; explicit fallback is disabled by default.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    PROVIDER_PREFLIGHT_ENABLED: bool = True
    PROVIDER_PREFLIGHT_TIMEOUT_SECONDS: float = Field(default=2, gt=0, le=10, allow_inf_nan=False)
    PROVIDER_FALLBACK_ENABLED: bool = False
    PROVIDER_FALLBACK_ALLOW_PAID: bool = False
    PROVIDER_FALLBACK_PROVIDER: LLMProvider | None = None
    PROVIDER_FALLBACK_MODEL_ID: ModelName | None = None
    OLLAMA_NUM_CTX: int = Field(default=16384, ge=2048, le=131072)
    GRAPH_TIMEOUT_SECONDS: float = Field(default=180, gt=0, allow_inf_nan=False)
    # None preserves the existing source-coverage, deficit and request deadlines.
    QUESTION_CANDIDATE_ATTEMPT_LIMIT: int | None = Field(default=None, ge=1, le=256)
    QUESTION_BATCH_RETRY_LIMIT: int | None = Field(default=None, ge=0, le=20)
    QUESTION_OVERALL_TIMEOUT_SECONDS: float | None = Field(
        default=None, gt=0, le=7200, allow_inf_nan=False
    )
    QA_HISTORY_MESSAGES: int = Field(default=12, ge=2)
    QA_HISTORY_CHARACTERS: int = Field(default=12000, ge=256)
    QA_RETENTION_MESSAGES: int = Field(default=40, ge=2)
    QA_RETENTION_CHARACTERS: int = Field(default=48000, ge=256)
    REVISION_HISTORY_LIMIT: int = Field(default=8, gt=0)
    REVISION_HISTORY_CHARACTERS: int = Field(default=12000, ge=1024)
    REVISION_CONTEXT_RECORDS: int = Field(default=3, gt=0)
    REVISION_CONTEXT_CHARACTERS: int = Field(default=6000, ge=512)
    REVISION_REQUIREMENTS_CHARACTERS: int = Field(default=8000, ge=512)
    AUTO_REVISION_LIMIT: int = Field(default=1, ge=0, le=3)
    REFINE_MODEL_CALL_LIMIT: int = Field(default=3, ge=1, le=6)
    CHECKPOINT_DATABASE: str = "runtime/checkpoints.sqlite"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    SESSION_RETENTION_DAYS: int = Field(default=0, ge=0, le=3650)
    SESSION_LEASE_SECONDS: float = Field(default=210, gt=0, allow_inf_nan=False)
    MAX_UPLOAD_BYTES: int = Field(default=268435456, ge=1024, le=2147483648)
    MAX_BATCH_FILES: int = Field(default=32, ge=1, le=128)
    MAX_BATCH_BYTES: int = Field(default=1073741824, ge=1024, le=8589934592)
    COMBINED_SEMANTIC_DEDUP_ENABLED: bool = True
    COMBINED_DEDUP_ALLOW_PAID: bool = False
    COMBINED_DEDUP_SIMILARITY: float = Field(default=0.94, ge=0.8, le=1, allow_inf_nan=False)
    COMBINED_DEDUP_MAX_RECORDS: int = Field(default=512, ge=2, le=2048)
    COMBINED_DEDUP_MAX_COMPARISONS: int = Field(default=16, ge=1, le=64)
    COMBINED_DEDUP_TIMEOUT_SECONDS: float = Field(default=15, gt=0, le=60, allow_inf_nan=False)
    INDEX_ON_UPLOAD: bool = False
    MEDIA_TIMEOUT_SECONDS: float = Field(default=180, gt=0, allow_inf_nan=False)
    VIDEO_MAX_DURATION_SECONDS: float = Field(default=7200, gt=0, allow_inf_nan=False)
    AUDIO_CHUNK_SECONDS: float = Field(default=300, ge=10, le=600)
    WHISPER_MODEL: ModelName = "tiny"
    WHISPER_LANGUAGE: str | None = None
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    WHISPER_NO_SPEECH_THRESHOLD: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    VISION_PROVIDER: Literal["disabled", "OLLAMA", "GOOGLE_GENAI"] = "OLLAMA"
    VISION_MODEL: str | None = "qwen3.5:2b"
    VISION_BASE_URL: str = "http://127.0.0.1:11434"
    VISION_TIMEOUT_SECONDS: float = Field(default=60, gt=0, allow_inf_nan=False)
    VLM_AUTOMATIC_MAX_VISUALS: int = Field(default=4, ge=1, le=128)
    VLM_MAX_VISUALS: int = Field(default=32, ge=1, le=128)
    VLM_MAX_CONCURRENCY: int = Field(default=2, ge=1, le=8)
    VLM_IMAGE_MAX_DIMENSION: int = Field(default=1280, ge=128, le=4096)
    VIDEO_FRAME_INTERVAL_SECONDS: float = Field(default=30, ge=1, le=3600, allow_inf_nan=False)
    VIDEO_MAX_FRAMES: int = Field(default=12, ge=1, le=64)
    VIDEO_SCENE_SCAN_FPS: float = Field(default=8, ge=2, le=16, allow_inf_nan=False)
    INGESTION_CHECKPOINT_RETENTION_HOURS: float = Field(
        default=24, ge=1, le=720, allow_inf_nan=False
    )
    VIDEO_FRAME_MAX_DIMENSION: int = Field(default=960, ge=128, le=2048)
    GENERATION_CONTEXT_CHARACTERS: int = Field(default=24000, ge=2000, le=200000)
    GENERATION_CHUNK_CHARACTERS: int = Field(default=12000, ge=1000, le=60000)
    GENERATION_MAX_CHUNKS: int = Field(default=64, ge=1, le=256)
    SEMANTIC_CONTENT_CLASSIFICATION_ENABLED: bool = True
    FFMPEG_BINARY: str | None = None
    FFPROBE_BINARY: str | None = None
    PDF_OCR_ENABLED: bool = False
    PDF_OCR_BACKEND: Literal["auto", "pymupdf", "tesseract"] = "auto"
    OCR_BINARY: str | None = None
    LIBREOFFICE_BINARY: str | None = None
    DOCX_RENDER_ENABLED: bool = True
    DOCX_RENDER_TIMEOUT_SECONDS: float = Field(default=45, gt=0, le=180)
    PDF_OCR_LANGUAGE: str = "eng"
    PDF_MAX_OCR_PAGES: int = Field(default=20, ge=1, le=100)
    FRONTEND_BACKEND_URL: str = "http://127.0.0.1:8000"
    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = Field(default=7860, ge=1, le=65535)
    FRONTEND_TIMEOUT_SECONDS: float = Field(default=600, gt=0, allow_inf_nan=False)
    FRONTEND_UPLOAD_TIMEOUT_SECONDS: float = Field(default=1800, gt=0, allow_inf_nan=False)
    FRONTEND_BROWSER_SECRET: SecretStr | None = None

    RAG_RETRIEVAL_STRATEGY: Literal["mmr", "similarity"] = "mmr"
    RAG_TOP_K: int = Field(default=5, ge=1, le=20)
    RAG_FETCH_K: int = Field(default=20, ge=1, le=100)
    RAG_MIN_RELEVANCE: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    RAG_MMR_LAMBDA: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    RAG_MAX_EVIDENCE_DOCUMENTS: int = Field(default=8, ge=1, le=20)
    RAG_MAX_EVIDENCE_CHARACTERS: int = Field(default=16000, ge=512)
    RAG_NEAR_DUPLICATE_THRESHOLD: float = Field(default=0.95, ge=0.8, le=1, allow_inf_nan=False)
    RAG_CONTEXT_WINDOW_TOKENS: int = Field(default=16384, ge=2048)
    RAG_PROMPT_RESERVE_TOKENS: int = Field(default=3000, ge=512)
    RAG_OUTPUT_RESERVE_TOKENS: int = Field(default=2048, ge=256)
    RAG_MAX_EVIDENCE_TOKENS: int = Field(default=7000, ge=256)
    RAG_HISTORY_MESSAGES: int = Field(default=6, ge=2)
    RAG_HISTORY_CHARACTERS: int = Field(default=2000, ge=256)
    RAG_REWRITE_ENABLED: bool = True
    RAG_QUERY_MAX_CHARACTERS: int = Field(default=1200, ge=128, le=2000)
    RAG_CORRECTIVE_PASSES: int = Field(default=1, ge=0, le=1)
    RAG_CORRECTIVE_TOP_K: int = Field(default=10, ge=1, le=20)
    RAG_CORRECTIVE_FETCH_K: int = Field(default=40, ge=1, le=100)

    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[1] / ".env")

    @field_validator(
        "QUESTION_GENERATER_PROVIDER",
        "QUESTION_REFINER_PROVIDER",
        "QUESTION_REWRITER_PROVIDER",
        "SUMMARIZER_MAINPOINT_PROVIDER",
        "SUMMARIZER_WRITER_PROVIDER",
        "SUMMARIZER_REWRITER_PROVIDER",
        "GRADE_QA_PROVIDER",
        "QUESTION_ANSWERER_PROVIDER",
        "EMBEDDING_MODEL_PROVIDER",
        "PROVIDER_FALLBACK_PROVIDER",
        "LLM_PROVIDER",
        mode="before",
    )
    @classmethod
    def normalize_provider(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator(
        "NVIDIA_NIM_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "LANGSMITH_API_KEY",
        "FRONTEND_BROWSER_SECRET",
        mode="before",
    )
    @classmethod
    def empty_secret(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator(
        "FFMPEG_BINARY",
        "FFPROBE_BINARY",
        "VISION_MODEL",
        "WHISPER_LANGUAGE",
        "OCR_BINARY",
        "LIBREOFFICE_BINARY",
        "PROVIDER_FALLBACK_PROVIDER",
        "PROVIDER_FALLBACK_MODEL_ID",
        "QUESTION_CANDIDATE_ATTEMPT_LIMIT",
        "QUESTION_BATCH_RETRY_LIMIT",
        "QUESTION_OVERALL_TIMEOUT_SECONDS",
        mode="before",
    )
    @classmethod
    def empty_optional_setting(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator(
        "OPENAI_COMPATIBLE_BASE_URL",
        "NVIDIA_NIM_BASE_URL",
        "GROQ_BASE_URL",
        "GOOGLE_GENAI_BASE_URL",
        "OLLAMA_BASE_URL",
        "VISION_BASE_URL",
    )
    @classmethod
    def validate_provider_base_url(cls, value, info):
        value = value.strip().rstrip("/") if isinstance(value, str) else value
        try:
            parsed = urlsplit(value) if isinstance(value, str) else None
            if parsed is not None:
                # Accessing port also validates malformed and out-of-range ports.
                parsed.port
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "?" in value
            or "#" in value
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(
                f"{info.field_name} must be an HTTP(S) URL without credentials, query, or fragment"
            )
        # Existing multimodal calls and metadata use Ollama's native /api paths.
        # Text transport adds /v1 when constructing its OpenAI-compatible client.
        if info.field_name == "OLLAMA_BASE_URL" or (
            info.field_name == "VISION_BASE_URL" and info.data.get("VISION_PROVIDER") == "OLLAMA"
        ):
            value = value.removesuffix("/v1")
        return value

    @field_validator("VISION_PROVIDER", mode="before")
    @classmethod
    def normalize_vision_provider(cls, value):
        if isinstance(value, str):
            return "disabled" if value.strip().lower() == "disabled" else value.strip().upper()
        return value

    @field_validator("WHISPER_LANGUAGE")
    @classmethod
    def validate_whisper_language(cls, value):
        if value is not None and (not value.isalpha() or len(value) > 32):
            raise ValueError("WHISPER_LANGUAGE must be a language name or code")
        return value.lower() if value else None

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("CHECKPOINT_DATABASE", mode="before")
    @classmethod
    def normalize_checkpoint_path(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, value, info):
        size = info.data.get("CHUNK_SIZE")
        if size is not None and value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def validate_history_budgets(self):
        if self.QA_RETENTION_MESSAGES < self.QA_HISTORY_MESSAGES:
            raise ValueError("QA_RETENTION_MESSAGES must cover QA_HISTORY_MESSAGES")
        if self.QA_RETENTION_CHARACTERS < self.QA_HISTORY_CHARACTERS:
            raise ValueError("QA_RETENTION_CHARACTERS must cover QA_HISTORY_CHARACTERS")
        if self.REVISION_HISTORY_LIMIT < self.REVISION_CONTEXT_RECORDS:
            raise ValueError("REVISION_HISTORY_LIMIT must cover REVISION_CONTEXT_RECORDS")
        if self.REVISION_HISTORY_CHARACTERS < self.REVISION_CONTEXT_CHARACTERS:
            raise ValueError("REVISION_HISTORY_CHARACTERS must cover REVISION_CONTEXT_CHARACTERS")
        if (
            self.RAG_FETCH_K < self.RAG_TOP_K
            or self.RAG_CORRECTIVE_FETCH_K < self.RAG_CORRECTIVE_TOP_K
        ):
            raise ValueError("RAG fetch limits must cover their selected chunk limits")
        available = (
            self.RAG_CONTEXT_WINDOW_TOKENS
            - self.RAG_PROMPT_RESERVE_TOKENS
            - self.RAG_OUTPUT_RESERVE_TOKENS
        )
        if available < 256 or self.RAG_MAX_EVIDENCE_TOKENS > available:
            raise ValueError("RAG context must reserve space for instructions, evidence and output")
        return self

    @model_validator(mode="after")
    def validate_runtime(self):
        if not self.CHECKPOINT_DATABASE:
            raise ValueError("CHECKPOINT_DATABASE is required")
        if self.SESSION_LEASE_SECONDS <= self.GRAPH_TIMEOUT_SECONDS:
            raise ValueError("SESSION_LEASE_SECONDS must be greater than GRAPH_TIMEOUT_SECONDS")
        # Vision is optional: incomplete provider setup disables the frontend
        # modes through vlm_policy without preventing ordinary text/audio work.
        if self.VLM_AUTOMATIC_MAX_VISUALS > self.VLM_MAX_VISUALS:
            raise ValueError("VLM_MAX_VISUALS must cover VLM_AUTOMATIC_MAX_VISUALS")
        return self


def get_settings():
    return Settings()


def print_env_variables():
    settings = get_settings()
    visible = settings.model_dump()
    logger = logging.getLogger("wizora.config")
    for key, value in visible.items():
        if isinstance(getattr(settings, key, None), SecretStr) or key.endswith(
            ("_API_KEY", "_TOKEN")
        ):
            value = "<set>" if value else "<not set>"
        logger.debug("%s=%s", key, value)
