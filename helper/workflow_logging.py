import logging
import re

from pydantic import SecretStr


_SECRET_FIELDS = (
    "GROQ_API_KEY",
    "GOOGLE_API_KEY",
    "HUGGING_FACE_HUB_TOKEN",
    "LANGSMITH_API_KEY",
    "OLLAMA_API_KEY",
)
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*\S+"
)


def secret_values(settings):
    values = []
    for name in _SECRET_FIELDS:
        value = getattr(settings, name, None)
        if isinstance(value, SecretStr):
            raw = value.get_secret_value().strip()
            if raw:
                values.append(raw)
    return values


def redact_text(text, secrets):
    if not isinstance(text, str):
        text = str(text)
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[redacted]")
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[redacted]", text)


class RedactFilter(logging.Filter):
    def __init__(self, secrets):
        super().__init__()
        self.secrets = tuple(secrets)

    def filter(self, record):
        record.msg = redact_text(record.getMessage(), self.secrets)
        record.args = ()
        return True


def configure_logging(settings):
    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    redactor = RedactFilter(secret_values(settings))
    root = logging.getLogger()
    root.setLevel(level)
    logging.getLogger("wizora").setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    for handler in root.handlers:
        if not any(isinstance(existing, RedactFilter) for existing in getattr(handler, "filters", ())):
            handler.addFilter(redactor)
