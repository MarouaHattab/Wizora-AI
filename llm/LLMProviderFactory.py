import math
import os
from collections.abc import Mapping

from pydantic import SecretStr
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from .Enums import LLMEnums


class CompatibleChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    @property
    def _supports_tool_choice(self) -> bool:
        return self.model.removeprefix("models/") == "gemini-3.8-flash" or super()._supports_tool_choice


class LLMProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str, model_id: str = None, model_temperature: float = 0.2):
        if not isinstance(provider, str) or provider.strip().upper() not in {item.value for item in LLMEnums}:
            raise ValueError("LLM provider must be OLLAMA, GOOGLE_GENAI, or GROQ")
        provider = provider.strip().upper()
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"A nonempty model_id is required for {provider}")
        model_id = model_id.strip()
        try:
            temperature = float(model_temperature)
        except (TypeError, ValueError):
            raise ValueError("Model temperature must be a finite non-negative number") from None
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Model temperature must be a finite non-negative number")
        if provider != LLMEnums.OLLAMA.value and temperature > 2:
            raise ValueError(f"Model temperature must be at most 2 for {provider}")

        if provider == LLMEnums.OLLAMA.value:
            return ChatOllama(model=model_id, temperature=temperature)
        elif provider == LLMEnums.GOOGLE_GENAI.value:
            identifier = model_id.removeprefix("models/").lower()
            if identifier.startswith(("gemini-1.0", "gemini-1.5", "gemini-2.0")):
                raise ValueError("The configured GOOGLE_GENAI model is retired; choose a supported model explicitly")
            return CompatibleChatGoogleGenerativeAI(
                model=model_id, temperature=temperature,
                google_api_key=self._credential("GOOGLE_API_KEY"),
            )
        elif provider == LLMEnums.GROQ.value:
            return ChatGroq(
                model=model_id, temperature=temperature,
                api_key=self._credential("GROQ_API_KEY"),
            )

    def _credential(self, name):
        value = self.config.get(name) if isinstance(self.config, Mapping) else getattr(self.config, name, None)
        if value is None:
            value = os.getenv(name)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required for the selected LLM provider")
        return value
