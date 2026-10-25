from importlib import import_module

_EXPORTS = {
    "BaseLLMProvider": "base",
    "NvidiaNimProvider": "nvidia_nim",
    "GroqProvider": "groq",
    "GoogleGenAIProvider": "google_genai",
    "OllamaProvider": "ollama",
    "OpenAICompatibleProvider": "compatible",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    result = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = result
    return result
