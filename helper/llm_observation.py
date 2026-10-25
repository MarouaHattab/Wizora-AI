from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import merge_configs

from chains import ProviderExecutionError, StructuredOutputError
from helper.workflow_metrics import metrics


class UsageCapture(BaseCallbackHandler):
    def __init__(self):
        self.chat_starts = 0
        self.llm_starts = 0
        self.tokens = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.chat_starts += 1

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_starts += 1

    def on_llm_end(self, response, **kwargs):
        self._usage(response)

    def on_chat_model_end(self, response, **kwargs):
        self._usage(response)

    @property
    def attempts(self):
        if self.chat_starts or self.llm_starts:
            return max(self.chat_starts, self.llm_starts)
        return None

    def _usage(self, response):
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage")
        if not isinstance(usage, dict):
            for group in getattr(response, "generations", ()) or ():
                for generation in group:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if isinstance(usage, dict):
                        break
                if isinstance(usage, dict):
                    break
        if not isinstance(usage, dict):
            return
        mapped = {}
        pairs = (
            ("input_tokens", ("input_tokens", "prompt_tokens")),
            ("output_tokens", ("output_tokens", "completion_tokens")),
            ("total_tokens", ("total_tokens",)),
        )
        for name, aliases in pairs:
            for alias in aliases:
                value = usage.get(alias)
                if type(value) is int and value >= 0:
                    mapped[name] = value
                    break
        self.tokens = mapped or None


class ObservedChain:
    def __init__(self, inner, *, task, provider, model):
        self.inner = inner
        self.task = task
        self.provider = provider
        self.model = model

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def invoke(self, value, config=None, **kwargs):
        started = perf_counter()
        usage = UsageCapture()
        try:
            result = self.inner.invoke(value, self._config(config, usage), **kwargs)
        except Exception as exc:
            self._record(started, "failed", usage, exc)
            raise
        self._record(started, "completed", usage, None)
        return result

    async def ainvoke(self, value, config=None, **kwargs):
        started = perf_counter()
        usage = UsageCapture()
        try:
            result = await self.inner.ainvoke(value, self._config(config, usage), **kwargs)
        except Exception as exc:
            self._record(started, "failed", usage, exc)
            raise
        self._record(started, "completed", usage, None)
        return result

    @staticmethod
    def _config(config, usage):
        return merge_configs(config, {"callbacks": [usage]})

    def _record(self, started, outcome, usage, error):
        category = "none"
        if isinstance(error, StructuredOutputError):
            category = "structured_output"
        elif isinstance(error, ProviderExecutionError):
            category = "provider"
        elif error is not None:
            category = type(error).__name__
        labels = {
            "task": self.task,
            "provider": self.provider,
            "model": self.model,
            "outcome": outcome,
        }
        metrics.increment("llm_invocations", **labels)
        metrics.observe("llm_latency_ms", (perf_counter() - started) * 1000, **labels)
        if usage.attempts is None:
            metrics.increment(
                "llm_attempts_unavailable",
                task=self.task,
                provider=self.provider,
                model=self.model,
            )
        else:
            metrics.increment(
                "llm_attempts",
                amount=usage.attempts,
                task=self.task,
                provider=self.provider,
                model=self.model,
            )
        if usage.tokens:
            metrics.add_tokens(
                task=self.task,
                provider=self.provider,
                model=self.model,
                tokens=usage.tokens,
            )
        else:
            metrics.increment(
                "llm_usage_unavailable",
                task=self.task,
                provider=self.provider,
                model=self.model,
            )
        if category != "none":
            metrics.increment(
                "llm_errors",
                task=self.task,
                provider=self.provider,
                model=self.model,
                category=category,
            )


def observe_chain(chain, *, task, provider, model):
    return ObservedChain(chain, task=task, provider=provider, model=model)
