import copy
import hashlib
import os
import time
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import RLock
from urllib.parse import quote, urlsplit

import httpx

TASK_PREFIXES = {
    "content_classification": "QUESTION_GENERATER",
    "question_generation": "QUESTION_GENERATER",
    "question_review": "QUESTION_REFINER",
    "question_revision": "QUESTION_REWRITER",
    "main_points": "SUMMARIZER_MAINPOINT",
    "summary": "SUMMARIZER_WRITER",
    "summary_revision": "SUMMARIZER_REWRITER",
    "summary_structure_revision": "SUMMARIZER_REWRITER",
    "topic_grade": "GRADE_QA",
    "answer": "QUESTION_ANSWERER",
    "query_rewrite": "QUESTION_ANSWERER",
    "evidence_assessment": "GRADE_QA",
    "grounded_answer": "QUESTION_ANSWERER",
    "grounding_check": "GRADE_QA",
    "vision": "VISION",
}
_CACHE = {}
_LOCK = RLock()
_AUDIT = ContextVar("wizora_provider_audit", default=None)


class ProviderUnavailableError(RuntimeError):
    pass


def setting(settings, name, default=None):
    return (
        settings.get(name, default)
        if isinstance(settings, Mapping)
        else getattr(settings, name, default)
    )


def _secret(settings, name):
    value = setting(settings, name)
    if value is None:
        value = os.getenv(name)
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return value.strip() if isinstance(value, str) else ""


def ollama_auth_headers(settings):
    credential = _secret(settings, "OLLAMA_API_KEY")
    return {"Authorization": "Bearer " + credential} if credential else {}


def clear_provider_cache():
    with _LOCK:
        _CACHE.clear()


def begin_provider_audit():
    return _AUDIT.set([])


def end_provider_audit(token):
    _AUDIT.reset(token)


def record_provider_resolution(resolution):
    audit = _AUDIT.get()
    if audit is not None:
        with _LOCK:
            audit.append(copy.deepcopy(resolution))
            del audit[:-256]


def provider_audit():
    with _LOCK:
        return copy.deepcopy(_AUDIT.get() or [])


def _prefix(task):
    if task in TASK_PREFIXES:
        return TASK_PREFIXES[task]
    if task in set(TASK_PREFIXES.values()):
        return task
    raise ValueError("Unknown provider task")


def _task_identity(task, settings):
    prefix = _prefix(task)
    if prefix == "VISION":
        return (
            str(setting(settings, "VISION_PROVIDER", "")).strip().upper(),
            str(setting(settings, "VISION_MODEL", "") or "").strip(),
        )
    model_name = prefix + "_MODEL_ID"
    return str(setting(settings, prefix + "_PROVIDER", "")).strip().upper(), str(
        setting(settings, model_name, "") or ""
    ).strip()


def _same_model(first, second):
    return str(first).removeprefix("models/") == str(second).removeprefix("models/")


def effective_model_info(task, settings):
    provider, model = _task_identity(task, settings)
    matches = [
        entry
        for entry in provider_audit()
        if entry.get("requested_provider") == provider
        and _same_model(entry.get("requested_model"), model)
    ]
    if matches:
        result = dict(matches[-1])
        result["executions"] = matches
        return result
    return {
        "provider": provider,
        "model": model,
        "requested_provider": provider,
        "requested_model": model,
        "fallback_used": False,
        "execution_verified": False,
    }


def _base_url(settings, vision=False, *, provider="OLLAMA"):
    name, default = {
        "OPENAI_COMPATIBLE": ("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
        "NVIDIA_NIM": ("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "GROQ": ("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "GOOGLE_GENAI": (
            "GOOGLE_GENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "OLLAMA": ("VISION_BASE_URL" if vision else "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    }[provider]
    value = str(setting(settings, name, default)).strip().rstrip("/")
    parsed = urlsplit(value)
    parsed.port
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("Invalid OpenAI-compatible endpoint")
    return value.removesuffix("/v1") if provider == "OLLAMA" else value


def _result(provider, model):
    return {
        "provider": provider,
        "model": model,
        "status": "ready",
        "available": True,
        "message": "Provider metadata is available; generation has not been tested.",
        "checks": {
            "reachable": None,
            "authentication": None,
            "model_available": None,
            "request_compatible": None,
        },
        "capabilities": {"text": None, "structured_output": None, "vision": None},
        "warnings": [],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "cached": False,
    }


def _failure(result, status, message):
    result.update(status=status, available=False, message=message)
    return result


def probe_provider(
    provider, model, settings, *, required_capabilities=("text",), refresh=False, vision=False
):
    provider = str(provider).strip().upper()
    model = str(model or "").strip()
    required = tuple(sorted(set(required_capabilities)))
    result = _result(provider, model)
    if (
        provider not in {"OPENAI_COMPATIBLE", "NVIDIA_NIM", "OLLAMA", "GOOGLE_GENAI", "GROQ"}
        or not model
    ):
        return _failure(
            result, "configuration_missing", "Choose a provider and a nonempty model identifier."
        )
    if any(cap not in result["capabilities"] for cap in required):
        raise ValueError("Unknown required provider capability")
    credential_name = {
        "OPENAI_COMPATIBLE": "OPENAI_COMPATIBLE_API_KEY",
        "NVIDIA_NIM": "NVIDIA_NIM_API_KEY",
        "GOOGLE_GENAI": "GOOGLE_API_KEY",
        "GROQ": "GROQ_API_KEY",
    }.get(provider, "OLLAMA_API_KEY")
    credential = _secret(settings, credential_name)
    if provider != "OLLAMA" and not credential:
        return _failure(result, "configuration_missing", credential_name + " is required.")
    try:
        endpoint = _base_url(settings, vision, provider=provider)
    except ValueError:
        return _failure(
            result,
            "configuration_invalid",
            "The provider endpoint must be an HTTP(S) URL without credentials, query or fragment.",
        )
    key = (provider, model, endpoint, hashlib.sha256(credential.encode()).hexdigest(), required)
    with _LOCK:
        cached = _CACHE.get(key)
        if not refresh and cached and cached[0] > time.monotonic():
            value = copy.deepcopy(cached[1])
            value["cached"] = True
            return value
    timeout = min(max(float(setting(settings, "PROVIDER_PREFLIGHT_TIMEOUT_SECONDS", 2)), 0.1), 10)
    try:
        with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            if provider in {"OPENAI_COMPATIBLE", "NVIDIA_NIM"}:
                headers = {"Authorization": "Bearer " + credential}
                response = client.get(endpoint + "/models", headers=headers)
                response.raise_for_status()
                result["checks"].update(reachable=True, authentication=True)
                names = [
                    entry.get("id", "")
                    for entry in response.json().get("data", [])
                    if isinstance(entry, dict)
                ]
                matches = model in names
                result["checks"]["model_available"] = matches
                if not matches:
                    _failure(
                        result,
                        "model_missing",
                        "The configured model is not available at the "
                        + ("NVIDIA NIM" if provider == "NVIDIA_NIM" else "OpenAI-compatible")
                        + " endpoint.",
                    )
                else:
                    result["capabilities"]["text"] = True
                    if provider == "NVIDIA_NIM":
                        # The adapter requests JSON in the prompt and validates it
                        # locally; catalog metadata cannot verify native constraints.
                        result["capabilities"]["structured_output"] = True
                        result["warnings"].append(
                            "Structured output uses prompt-based JSON with local schema validation; "
                            "native constrained-output and vision support are unverified in metadata. "
                            "Generation has not been tested."
                        )
                    else:
                        result["warnings"].append(
                            "Structured-output and vision support are provider/model-specific and require request validation."
                        )
            elif provider == "OLLAMA":
                headers = {"Authorization": "Bearer " + credential} if credential else {}
                response = client.get(endpoint + "/api/tags", headers=headers)
                response.raise_for_status()
                result["checks"].update(reachable=True, authentication=True)
                installed = response.json().get("models", [])
                names = [
                    entry.get("name", entry.get("model", ""))
                    for entry in installed
                    if isinstance(entry, dict)
                ]
                matches = model in names or (":" not in model and model + ":latest" in names)
                result["checks"]["model_available"] = matches
                if not matches:
                    _failure(
                        result,
                        "model_missing",
                        "The configured model is not installed at the Ollama endpoint. Pull it explicitly or choose an installed model.",
                    )
                else:
                    response = client.post(
                        endpoint + "/api/show", headers=headers, json={"model": model}
                    )
                    response.raise_for_status()
                    capabilities = response.json().get("capabilities")
                    if isinstance(capabilities, list):
                        # JSON-schema output uses Ollama's format parameter, not
                        # tool calling. Completion-only models can use it too.
                        result["capabilities"].update(
                            text="completion" in capabilities,
                            structured_output="completion" in capabilities,
                            vision="vision" in capabilities,
                        )
                    else:
                        result["warnings"].append(
                            "This Ollama version does not report model capabilities; request support is unverified."
                        )
            elif provider == "GOOGLE_GENAI":
                identifier = model.removeprefix("models/")
                response = client.get(
                    endpoint.removesuffix("/openai") + "/models/" + quote(identifier, safe=""),
                    headers={"x-goog-api-key": credential},
                )
                response.raise_for_status()
                payload = response.json()
                result["checks"].update(reachable=True, authentication=True, model_available=True)
                methods = payload.get("supportedGenerationMethods")
                if isinstance(methods, list):
                    result["capabilities"]["text"] = "generateContent" in methods
                # Google's model metadata does not currently declare image/tool support.
                result["warnings"].append(
                    "Image and structured-output support are not declared by this model metadata endpoint; an actual request can still fail."
                )
            else:
                endpoint += "/models"
                response = client.get(
                    endpoint if "/" in model else endpoint + "/" + quote(model, safe=""),
                    headers={"Authorization": "Bearer " + credential},
                )
                response.raise_for_status()
                payload = response.json()
                if "/" in model:
                    payload = next(
                        (item for item in payload.get("data", []) if item.get("id") == model), None
                    )
                    if payload is None:
                        payload = {"active": False}
                        _failure(
                            result,
                            "model_missing",
                            "The configured model is not available on this Groq connection.",
                        )
                result["checks"].update(
                    reachable=True,
                    authentication=True,
                    model_available=payload.get("active", True) is not False,
                )
                result["capabilities"]["text"] = payload.get("active", True) is not False
                result["warnings"].append(
                    "Image and structured-output support are not declared by this model metadata endpoint; an actual request can still fail."
                )
            if result["available"]:
                supported = [result["capabilities"][cap] for cap in required]
                result["checks"]["request_compatible"] = (
                    False if False in supported else None if None in supported else True
                )
                if False in supported:
                    missing = ", ".join(
                        cap for cap in required if result["capabilities"][cap] is False
                    )
                    _failure(
                        result,
                        "request_incompatible"
                        if "text" in required and result["capabilities"]["text"] is False
                        else "capability_missing",
                        "The configured model metadata does not support the requested capability: "
                        + missing
                        + ".",
                    )
                elif None in supported:
                    result["status"] = "available_unverified"
                    result["warnings"].append(
                        "Required capability support is unknown in metadata; no inference was performed."
                    )
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        result["checks"]["reachable"] = True
        if status in {401, 403}:
            result["checks"]["authentication"] = False
            _failure(
                result,
                "authentication_failed",
                "Provider authentication failed. Check the configured key and its permissions.",
            )
        elif status == 404:
            result["checks"]["model_available"] = False
            _failure(
                result,
                "model_missing",
                "The configured model or metadata endpoint was not found. Check the model identifier and service version.",
            )
        elif status == 429:
            _failure(
                result,
                "rate_limited",
                "The provider rate limit or quota prevents verification. Retry after the provider allows requests.",
            )
        else:
            _failure(
                result,
                "service_error",
                "The provider metadata service returned an error. Retry or inspect the provider service.",
            )
    except (httpx.HTTPError, httpx.InvalidURL):
        result["checks"]["reachable"] = False
        _failure(
            result,
            "offline",
            "The configured provider cannot be reached. Start the service and check its endpoint or connection.",
        )
    except (ValueError, TypeError, AttributeError):
        _failure(
            result,
            "metadata_invalid",
            "The provider returned unsupported model metadata. Check its endpoint and service version.",
        )
    with _LOCK:
        if len(_CACHE) >= 128:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (time.monotonic() + (30 if result["available"] else 3), copy.deepcopy(result))
    return result


def resolve_provider(
    provider, model, settings, *, required_capabilities=("text",), refresh=False, vision=False
):
    status = probe_provider(
        provider,
        model,
        settings,
        required_capabilities=required_capabilities,
        refresh=refresh,
        **({"vision": True} if vision else {}),
    )
    resolution = {
        "requested_provider": provider,
        "requested_model": model,
        "provider": provider,
        "model": model,
        "fallback_used": False,
        "fallback_reason": None,
        "preflight_status": status["status"],
        "execution_verified": False,
    }
    if status["available"]:
        return resolution
    fallback_provider = str(setting(settings, "PROVIDER_FALLBACK_PROVIDER", "") or "").upper()
    fallback_model = str(setting(settings, "PROVIDER_FALLBACK_MODEL_ID", "") or "").strip()
    if (
        not setting(settings, "PROVIDER_FALLBACK_ENABLED", False)
        or not fallback_provider
        or not fallback_model
    ):
        raise ProviderUnavailableError(status["message"])
    if fallback_provider != "OLLAMA" and not setting(
        settings, "PROVIDER_FALLBACK_ALLOW_PAID", False
    ):
        raise ProviderUnavailableError(
            status["message"] + " Paid fallback is disabled in settings."
        )
    if fallback_provider == provider and _same_model(fallback_model, model):
        raise ProviderUnavailableError(status["message"])
    fallback = probe_provider(
        fallback_provider,
        fallback_model,
        settings,
        required_capabilities=required_capabilities,
        refresh=refresh,
    )
    if not fallback["available"]:
        raise ProviderUnavailableError(
            status["message"]
            + " The explicitly configured fallback is also unavailable: "
            + fallback["message"]
        )
    resolution.update(
        provider=fallback_provider,
        model=fallback_model,
        fallback_used=True,
        fallback_reason=status["status"],
        preflight_status=fallback["status"],
    )
    return resolution


def resolve_task_provider(task, settings, *, required_capabilities=("text",), refresh=False):
    provider, model = _task_identity(task, settings)
    return resolve_provider(
        provider,
        model,
        settings,
        required_capabilities=required_capabilities,
        refresh=refresh,
        **({"vision": True} if _prefix(task) == "VISION" else {}),
    )


def provider_status(settings, *, refresh=False):
    entries = []
    probes = {}
    # A forced refresh bypasses stale global entries once per distinct request.
    for prefix in dict.fromkeys(TASK_PREFIXES.values()):
        provider, model = _task_identity(prefix, settings)
        if prefix == "VISION" and provider == "DISABLED":
            entries.append(
                {
                    "task": "VISION",
                    **_failure(
                        _result(provider, model),
                        "disabled",
                        "Visual understanding is disabled in settings.",
                    ),
                }
            )
            continue
        required = (
            ("vision",)
            if prefix == "VISION"
            else ("text", "structured_output")
            if prefix in {"QUESTION_GENERATER", "QUESTION_REFINER", "QUESTION_REWRITER", "GRADE_QA"}
            else ("text",)
        )
        key = (provider, model, required, prefix == "VISION")
        if key not in probes:
            probes[key] = probe_provider(
                provider,
                model,
                settings,
                required_capabilities=required,
                refresh=refresh,
                vision=prefix == "VISION",
            )
        entries.append({"task": prefix, **copy.deepcopy(probes[key])})
    return {
        "providers": entries,
        "metadata_only": True,
        "fallback": {
            "enabled": bool(setting(settings, "PROVIDER_FALLBACK_ENABLED", False)),
            "allow_paid": bool(setting(settings, "PROVIDER_FALLBACK_ALLOW_PAID", False)),
            "provider": setting(settings, "PROVIDER_FALLBACK_PROVIDER"),
            "model": setting(settings, "PROVIDER_FALLBACK_MODEL_ID"),
        },
    }
