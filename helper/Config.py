from typing import Annotated, Literal

from pydantic import Field, SecretStr, StringConstraints, field_validator
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

    model_config = SettingsConfigDict(env_file=".env")

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
        "GROQ_API_KEY", "GOOGLE_API_KEY", "HUGGING_FACE_HUB_TOKEN", "LANGSMITH_API_KEY",
        mode="before",
    )
    @classmethod
    def empty_secret(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, value, info):
        size = info.data.get("CHUNK_SIZE")
        if size is not None and value >= size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

def get_settings():
    return Settings()

def print_env_variables():
    settings = get_settings()
    for key, value in settings:
        if isinstance(value, SecretStr) or key.endswith(("_API_KEY", "_TOKEN")):
            value = "<set>" if value else "<not set>"
        print(f"{key}: {value}")
