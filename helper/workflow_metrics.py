from threading import Lock


def _label(value):
    text = str(value)
    cleaned = "".join(
        char if char.isalnum() or char in {"_", "-", ".", "/"} else "_" for char in text
    )
    return cleaned[:80] or "unknown"


class WorkflowMetrics:
    def __init__(self):
        self._lock = Lock()
        self._counters = {}
        self._observations = {}

    def increment(self, name, amount=1, **labels):
        if type(amount) is not int or amount < 0:
            return
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def observe(self, name, value, **labels):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if number < 0:
            return
        key = self._key(name, labels)
        with self._lock:
            current = self._observations.setdefault(key, {"count": 0, "sum": 0.0})
            current["count"] += 1
            current["sum"] = round(current["sum"] + number, 3)

    def add_tokens(self, *, task, provider, model, tokens):
        self.increment("llm_usage_reports", task=task, provider=provider, model=model)
        for name, value in tokens.items():
            self.increment(f"llm_{name}", amount=value, task=task, provider=provider, model=model)

    def snapshot(self):
        with self._lock:
            observations = {
                key: {"count": value["count"], "sum": value["sum"]}
                for key, value in self._observations.items()
            }
            return {"counters": dict(self._counters), "observations": observations}

    @staticmethod
    def _key(name, labels):
        if not labels:
            return name
        suffix = ",".join(f"{_label(key)}={_label(labels[key])}" for key in sorted(labels))
        return f"{name}|{suffix}"


metrics = WorkflowMetrics()
