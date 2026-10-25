from typing import ClassVar

from .base import BaseLLMProvider


class GoogleGenAIProvider(BaseLLMProvider):
    provider_name: ClassVar[str] = "GOOGLE_GENAI"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    native_json: ClassVar[bool] = True

    def _structured_parameters(self, schema):
        return self._schema_parameters(schema)
