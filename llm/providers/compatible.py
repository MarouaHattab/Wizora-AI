from .base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    base_url: str = "https://api.openai.com/v1"
