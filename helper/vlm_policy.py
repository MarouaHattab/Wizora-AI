import time
from collections.abc import Mapping
from threading import Lock
from typing import Literal

import httpx

VLMMode = Literal["disabled", "automatic", "enhanced"]
VLM_MODES = ("disabled", "automatic", "enhanced")
_CAPABILITY_CACHE = {}
_CAPABILITY_LOCK = Lock()


def _value(settings, name, default=None):
    return (
        settings.get(name, default)
        if isinstance(settings, Mapping)
        else getattr(settings, name, default)
    )


def normalize_vlm_mode(value) -> VLMMode:
    if not isinstance(value, str) or value not in VLM_MODES:
        raise ValueError("vlm_mode must be disabled, automatic, or enhanced")
    return value


def vlm_unavailable_reason(settings) -> str:
    provider = _value(settings, "VISION_PROVIDER", "disabled")
    if provider not in {"OPENAI_COMPATIBLE", "OLLAMA", "GOOGLE_GENAI"}:
        return "Visual understanding is unavailable because no VLM provider is configured. Text, tables and audio still work."
    model = _value(settings, "VISION_MODEL")
    if not isinstance(model, str) or not model.strip():
        return "Visual understanding is unavailable because no vision model is configured. Text, tables and audio still work."
    if provider == "GOOGLE_GENAI":
        key = _value(settings, "GOOGLE_API_KEY")
        key = key.get_secret_value() if hasattr(key, "get_secret_value") else key
        if not isinstance(key, str) or not key.strip():
            return "Visual understanding is unavailable because the configured VLM provider has no API key. Text, tables and audio still work."
    return ""


def vlm_available(settings) -> bool:
    return not vlm_unavailable_reason(settings)


def resolve_vlm_mode(value, settings, source_type=None) -> VLMMode:
    mode = normalize_vlm_mode(value)
    if mode == "disabled" or source_type in {"text", "audio"} or not vlm_available(settings):
        return "disabled"
    return mode


def vlm_capabilities(settings) -> dict:
    available = vlm_available(settings)
    return {
        "vlm_available": available,
        "vlm_default_mode": "automatic" if available else "disabled",
        "vlm_modes": list(VLM_MODES) if available else ["disabled"],
        "vlm_unavailable_reason": vlm_unavailable_reason(settings),
    }


def frontend_vlm_capabilities(settings) -> dict:
    capabilities = vlm_capabilities(settings)
    if not capabilities["vlm_available"] or _value(settings, "VISION_PROVIDER") != "OLLAMA":
        return capabilities
    model = _value(settings, "VISION_MODEL").strip()
    base_url = str(_value(settings, "VISION_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    import hashlib

    from helper.provider_status import ollama_auth_headers

    headers = ollama_auth_headers(settings)
    key = (base_url, model, hashlib.sha256(headers.get("Authorization", "").encode()).hexdigest())
    with _CAPABILITY_LOCK:
        cached = _CAPABILITY_CACHE.get(key)
        if cached and cached[0] > time.monotonic():
            return dict(cached[1])
        reason = ""
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                response = client.post(
                    base_url + "/api/show", headers=headers, json={"model": model}
                )
                response.raise_for_status()
                supported = response.json().get("capabilities", [])
            if not isinstance(supported, list) or "vision" not in supported:
                reason = "The configured Ollama model does not support image inputs. Choose a vision-capable model; text, tables and audio still work."
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError, AttributeError):
            reason = "The configured Ollama vision service or model could not be verified. Start the service and check the model; text, tables and audio still work."
        if reason:
            capabilities.update(
                vlm_available=False,
                vlm_default_mode="disabled",
                vlm_modes=["disabled"],
                vlm_unavailable_reason=reason,
            )
        if len(_CAPABILITY_CACHE) >= 64:
            _CAPABILITY_CACHE.pop(next(iter(_CAPABILITY_CACHE)))
        _CAPABILITY_CACHE[key] = (time.monotonic() + (10 if reason else 60), dict(capabilities))
        return capabilities
