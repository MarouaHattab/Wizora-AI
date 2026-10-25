from typing import ClassVar

from .base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    provider_name: ClassVar[str] = "GROQ"
    base_url: str = "https://api.groq.com/openai/v1"
    native_json: ClassVar[bool] = True

    def _structured_parameters(self, schema):
        if self.model_name in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            return self._schema_parameters(schema)
        return super()._structured_parameters(schema)
