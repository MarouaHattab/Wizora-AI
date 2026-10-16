import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Temperature = Annotated[float, Field(ge=0, allow_inf_nan=False)]
LLMProvider = Literal["OLLAMA", "GOOGLE_GENAI", "GROQ"]
EmbeddingProvider = Literal["LOCAL_EMBEDDING", "HUGGINGFACE"]

class Settings(BaseSettings): 
    # API keys and tokens
    GROQ_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
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
    OLLAMA_NUM_CTX: int = Field(default=16384, ge=2048, le=131072)
    GRAPH_TIMEOUT_SECONDS: float = Field(default=180, gt=0, allow_inf_nan=False)
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
    INDEX_ON_UPLOAD: bool = False
    MEDIA_TIMEOUT_SECONDS: float = Field(default=180, gt=0, allow_inf_nan=False)
    VIDEO_MAX_DURATION_SECONDS: float = Field(default=7200, gt=0, allow_inf_nan=False)
    AUDIO_CHUNK_SECONDS: float = Field(default=300, ge=10, le=600)
    FFMPEG_BINARY: str | None = None
    FFPROBE_BINARY: str | None = None
    PDF_OCR_ENABLED: bool = False
    PDF_OCR_LANGUAGE: str = "eng"
    PDF_MAX_OCR_PAGES: int = Field(default=20, ge=1, le=100)
    FRONTEND_BACKEND_URL: str = "http://127.0.0.1:8000"
    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = Field(default=7860, ge=1, le=65535)
    FRONTEND_TIMEOUT_SECONDS: float = Field(default=600, gt=0, allow_inf_nan=False)
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
        "QUESTION_GENERATER_PROVIDER", "QUESTION_REFINER_PROVIDER",
        "QUESTION_REWRITER_PROVIDER", "SUMMARIZER_MAINPOINT_PROVIDER",
        "SUMMARIZER_WRITER_PROVIDER", "SUMMARIZER_REWRITER_PROVIDER",
        "GRADE_QA_PROVIDER", "QUESTION_ANSWERER_PROVIDER", "EMBEDDING_MODEL_PROVIDER",
        mode="before",
    )
    @classmethod
    def normalize_provider(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator(
        "GROQ_API_KEY", "GOOGLE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "LANGSMITH_API_KEY", "FRONTEND_BROWSER_SECRET",
        mode="before",
    )
    @classmethod
    def empty_secret(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("FFMPEG_BINARY", "FFPROBE_BINARY", mode="before")
    @classmethod
    def empty_optional_setting(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

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
        if self.RAG_FETCH_K < self.RAG_TOP_K or self.RAG_CORRECTIVE_FETCH_K < self.RAG_CORRECTIVE_TOP_K:
            raise ValueError("RAG fetch limits must cover their selected chunk limits")
        available = self.RAG_CONTEXT_WINDOW_TOKENS - self.RAG_PROMPT_RESERVE_TOKENS - self.RAG_OUTPUT_RESERVE_TOKENS
        if available < 256 or self.RAG_MAX_EVIDENCE_TOKENS > available:
            raise ValueError("RAG context must reserve space for instructions, evidence and output")
        return self

    @model_validator(mode="after")
    def validate_runtime(self):
        if not self.CHECKPOINT_DATABASE:
            raise ValueError("CHECKPOINT_DATABASE is required")
        if self.SESSION_LEASE_SECONDS <= self.GRAPH_TIMEOUT_SECONDS:
            raise ValueError("SESSION_LEASE_SECONDS must be greater than GRAPH_TIMEOUT_SECONDS")
        return self

def get_settings():
    return Settings()

def print_env_variables():
    settings = get_settings()
    visible = settings.model_dump()
    logger = logging.getLogger("wizora.config")
    for key, value in visible.items():
        if isinstance(getattr(settings, key, None), SecretStr) or key.endswith(("_API_KEY", "_TOKEN")):
            value = "<set>" if value else "<not set>"
        logger.debug("%s=%s", key, value)
