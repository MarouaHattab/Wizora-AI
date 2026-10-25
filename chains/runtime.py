import asyncio
import hashlib
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock

from langchain_core.runnables import Runnable
from pydantic import SecretStr

from chains.contracts import ChainInputError, ProviderExecutionError
from helper import provider_status
from helper.llm_observation import observe_chain
from llm import LLMProviderFactory

from .builders import build_task_chain

TASK_CONFIGURATION = {
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
}
_MODEL_LIMIT = 16
_CHAIN_LIMIT = 64
_models = OrderedDict()
_chains = OrderedDict()
_lock = RLock()


class PreflightModel(Runnable):
    def __init__(self, settings, provider, model_id, temperature, timeout, *, schema=None):
        self.settings = settings
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.timeout = timeout
        self.schema = schema
        self.custom_get_token_ids = None
        self._clients = OrderedDict()
        self._client_lock = RLock()

    def with_structured_output(self, schema):
        return PreflightModel(
            self.settings,
            self.provider,
            self.model_id,
            self.temperature,
            self.timeout,
            schema=schema,
        )

    def _resolve(self):
        required = ("text", "structured_output") if self.schema is not None else ("text",)
        return provider_status.resolve_provider(
            self.provider, self.model_id, self.settings, required_capabilities=required
        )

    def _client(self, resolution):
        key = (resolution["provider"], resolution["model"], _scope())
        with self._client_lock:
            for old_key in list(self._clients):
                loop = old_key[2][1]
                if loop is not None and loop.is_closed():
                    del self._clients[old_key]
            client = self._clients.get(key)
            if client is None:
                client = LLMProviderFactory(self.settings).create(
                    key[0], key[1], self.temperature, self.timeout
                )
                if self.schema is not None:
                    client = client.with_structured_output(self.schema)
                _remember(self._clients, key, client, 8)
            return client

    @staticmethod
    def _record(resolution, outcome):
        provider_status.record_provider_resolution(
            {**resolution, "execution_verified": True, "outcome": outcome}
        )

    def invoke(self, value, config=None, **kwargs):
        resolution = self._resolve()
        try:
            result = self._client(resolution).invoke(value, config=config, **kwargs)
        except BaseException as error:
            self._record(
                resolution,
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else "failed",
            )
            raise
        self._record(resolution, "completed")
        return result

    async def ainvoke(self, value, config=None, **kwargs):
        resolution = await asyncio.to_thread(self._resolve)
        try:
            result = await self._client(resolution).ainvoke(value, config=config, **kwargs)
        except BaseException as error:
            self._record(
                resolution,
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else "failed",
            )
            raise
        self._record(resolution, "completed")
        return result

    def stream(self, value, config=None, **kwargs):
        resolution = self._resolve()
        try:
            yield from self._client(resolution).stream(value, config=config, **kwargs)
        except BaseException as error:
            self._record(
                resolution,
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else "failed",
            )
            raise
        self._record(resolution, "completed")

    async def astream(self, value, config=None, **kwargs):
        resolution = await asyncio.to_thread(self._resolve)
        try:
            async for chunk in self._client(resolution).astream(value, config=config, **kwargs):
                yield chunk
        except BaseException as error:
            self._record(
                resolution,
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, GeneratorExit))
                else "failed",
            )
            raise
        self._record(resolution, "completed")


def token_counter_for(llm):
    counter = getattr(llm, "custom_get_token_ids", None)
    if not callable(counter):
        return None

    def count(text):
        if not isinstance(text, str):
            raise ChainInputError("Token counting requires text")
        return len(counter(text))

    return count


def _setting(settings, name, default=None):
    return (
        settings.get(name, default)
        if isinstance(settings, Mapping)
        else getattr(settings, name, default)
    )


def _scope():
    try:
        return ("async", asyncio.get_running_loop())
    except RuntimeError:
        return ("sync", None)


def _credential_fingerprint(settings, provider):
    name = {
        "OPENAI_COMPATIBLE": "OPENAI_COMPATIBLE_API_KEY",
        "NVIDIA_NIM": "NVIDIA_NIM_API_KEY",
        "GOOGLE_GENAI": "GOOGLE_API_KEY",
        "GROQ": "GROQ_API_KEY",
        "OLLAMA": "OLLAMA_API_KEY",
    }.get(provider)
    if name is None:
        return ""
    secret = _setting(settings, name)
    if secret is None:
        secret = os.getenv(name)
    if isinstance(secret, SecretStr):
        secret = secret.get_secret_value()
    if not isinstance(secret, str):
        return ""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _provider_configuration(settings):
    endpoints = (
        ("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
        ("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        ("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        ("GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    return (
        tuple(str(_setting(settings, name, default)) for name, default in endpoints),
        tuple(
            _credential_fingerprint(settings, provider)
            for provider in (
                "OPENAI_COMPATIBLE",
                "NVIDIA_NIM",
                "GROQ",
                "GOOGLE_GENAI",
                "OLLAMA",
            )
        ),
        _setting(settings, "LLM_MAX_RETRIES", 2),
        _setting(settings, "LLM_MAX_CONCURRENCY", 2),
        _setting(settings, "LLM_MAX_OUTPUT_TOKENS", 2048),
        _setting(settings, "LLM_RETRY_MAX_WAIT_SECONDS", 10),
    )


def _discard_closed_loops():
    for key in list(_models):
        if key[0][1] is not None and key[0][1].is_closed():
            del _models[key]
    for key in list(_chains):
        if key[1][0][1] is not None and key[1][0][1].is_closed():
            del _chains[key]


def _model_key(task, settings):
    if not isinstance(task, str) or task not in TASK_CONFIGURATION:
        raise ChainInputError("Unknown LLM task")
    prefix = TASK_CONFIGURATION[task]
    provider = _setting(settings, f"{prefix}_PROVIDER")
    model_id = _setting(settings, f"{prefix}_MODEL_ID")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        raise ChainInputError("Task model configuration is incomplete")
    provider, model_id = provider.strip().upper(), model_id.strip()
    if not provider or not model_id:
        raise ChainInputError("Task model configuration is incomplete")
    try:
        temperature = float(_setting(settings, f"{prefix}_TEMPERATURE"))
        timeout = float(_setting(settings, "LLM_TIMEOUT_SECONDS", 60))
    except (TypeError, ValueError) as exc:
        raise ChainInputError("Task model configuration is invalid") from exc
    if (
        not math.isfinite(temperature)
        or temperature < 0
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ChainInputError("Task model configuration is invalid")
    if provider == "GOOGLE_GENAI" and not model_id.startswith(("models/", "tunedModels/")):
        model_id = f"models/{model_id}"
    return (
        _scope(),
        provider,
        model_id,
        temperature,
        timeout,
        _credential_fingerprint(settings, provider),
        _setting(settings, "OLLAMA_NUM_CTX", 16384) if provider == "OLLAMA" else None,
        (
            bool(_setting(settings, "PROVIDER_PREFLIGHT_ENABLED", False)),
            str(_setting(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            bool(_setting(settings, "PROVIDER_FALLBACK_ENABLED", False)),
            bool(_setting(settings, "PROVIDER_FALLBACK_ALLOW_PAID", False)),
            _setting(settings, "PROVIDER_FALLBACK_PROVIDER"),
            _setting(settings, "PROVIDER_FALLBACK_MODEL_ID"),
            _provider_configuration(settings),
        ),
    )


def _resolve_model(settings, model_key):
    model = _models.get(model_key)
    if model is None:
        _, provider, model_id, temperature, timeout, _, _, options = model_key
        try:
            model = (
                PreflightModel(settings, provider, model_id, temperature, timeout)
                if options[0] or options[2]
                else LLMProviderFactory(settings).create(provider, model_id, temperature, timeout)
            )
        except Exception as exc:
            raise ProviderExecutionError("The configured model could not be initialized") from exc
        _remember(_models, model_key, model, _MODEL_LIMIT)
    else:
        _models.move_to_end(model_key)
    return model


def _input_budget(settings, model_key):
    configured = int(_setting(settings, "GENERATION_CONTEXT_CHARACTERS", 24000))
    if model_key[1] == "OLLAMA" or (
        _setting(settings, "PROVIDER_FALLBACK_ENABLED", False)
        and _setting(settings, "PROVIDER_FALLBACK_PROVIDER") == "OLLAMA"
    ):
        configured = min(configured, int(_setting(settings, "OLLAMA_NUM_CTX", 16384)) - 2048)
    return configured


def get_task_chain(task: str, settings, *, llm=None) -> Runnable:
    if not isinstance(task, str) or task not in TASK_CONFIGURATION:
        raise ChainInputError("Unknown LLM task")
    if llm is not None:
        return build_task_chain(task, llm)
    model_key = _model_key(task, settings)
    budget = _input_budget(settings, model_key)
    chain_key = (task, model_key, budget)
    with _lock:
        _discard_closed_loops()
        if chain_key in _chains:
            _chains.move_to_end(chain_key)
            return _chains[chain_key][0]
        model = _resolve_model(settings, model_key)
        _, provider, model_id, _, _, _, _, _ = model_key
        chain = observe_chain(
            build_task_chain(task, model).with_config(
                configurable={"generation_input_budget": budget}
            ),
            task=task,
            provider=provider,
            model=model_id,
        )
        _remember(_chains, chain_key, (chain, model), _CHAIN_LIMIT)
        return chain


def get_task_token_counter(task: str, settings, *, llm=None, initialize=True):
    if not isinstance(task, str) or task not in TASK_CONFIGURATION:
        raise ChainInputError("Unknown LLM task")
    if llm is not None:
        return token_counter_for(llm)
    model_key = _model_key(task, settings)
    chain_key = (task, model_key, _input_budget(settings, model_key))
    with _lock:
        _discard_closed_loops()
        if chain_key in _chains:
            _chains.move_to_end(chain_key)
            return token_counter_for(_chains[chain_key][1])
        if not initialize:
            return None
        return token_counter_for(_resolve_model(settings, model_key))
