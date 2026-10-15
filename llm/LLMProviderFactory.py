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

    def _request_options(self, kwargs):
        # The pinned adapter declares timeout without forwarding it to its SDK.
        return {"timeout": self.timeout, "retry": None, **kwargs}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop=stop, run_manager=run_manager,
                                 **self._request_options(kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager,
                                        **self._request_options(kwargs))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield from super()._stream(messages, stop=stop, run_manager=run_manager,
                                   **self._request_options(kwargs))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager,
                                            **self._request_options(kwargs)):
            yield chunk


class LLMProviderFactory:
    def __init__(self, config):
        self.config = config

    def create(self, provider: str, model_id: str = None, model_temperature: float = 0.2,
               timeout_seconds: float | None = None):
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
        if timeout_seconds is None:
            timeout_seconds = self._setting("LLM_TIMEOUT_SECONDS", 60)
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError):
            raise ValueError("LLM timeout must be a finite positive number") from None
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("LLM timeout must be a finite positive number")

        if provider == LLMEnums.OLLAMA.value:
            return ChatOllama(model=model_id, temperature=temperature,
                              client_kwargs={"timeout": timeout_seconds})
        elif provider == LLMEnums.GOOGLE_GENAI.value:
            identifier = model_id.removeprefix("models/").lower()
            if identifier.startswith(("gemini-1.0", "gemini-1.5", "gemini-2.0")):
                raise ValueError("The configured GOOGLE_GENAI model is retired; choose a supported model explicitly")
            return CompatibleChatGoogleGenerativeAI(
                model=model_id, temperature=temperature,
                google_api_key=self._credential("GOOGLE_API_KEY"),
                timeout=timeout_seconds,
            )
        elif provider == LLMEnums.GROQ.value:
            return ChatGroq(
                model=model_id, temperature=temperature,
                api_key=self._credential("GROQ_API_KEY"),
                timeout=timeout_seconds,
            )

    def _setting(self, name, default=None):
        return self.config.get(name, default) if isinstance(self.config, Mapping) else getattr(self.config, name, default)

    def _credential(self, name):
        value = self._setting(name)
        if value is None:
            value = os.getenv(name)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required for the selected LLM provider")
        return value
