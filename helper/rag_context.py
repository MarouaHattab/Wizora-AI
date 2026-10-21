import json
import re
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from chains.QuestionAnswerChain import prepare_conversation_history
from chains.contracts import ChainInputError


def bounded_history(messages, question, settings):
    return prepare_conversation_history(
        messages, question,
        history_messages=min(settings.RAG_HISTORY_MESSAGES, settings.QA_HISTORY_MESSAGES),
        history_characters=min(settings.RAG_HISTORY_CHARACTERS, settings.QA_HISTORY_CHARACTERS),
    )


def bounded_user_history(messages, settings):
    selected, remaining = [], min(settings.RAG_HISTORY_CHARACTERS, settings.QA_HISTORY_CHARACTERS)
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if len(message.content) > remaining:
            break
        selected.append(message)
        remaining -= len(message.content)
        if len(selected) >= min(settings.RAG_HISTORY_MESSAGES, settings.QA_HISTORY_MESSAGES):
            break
    return list(reversed(selected))


def needs_context(question, history):
    if not any(isinstance(message, HumanMessage) for message in history):
        return False
    question = question.strip()
    if re.match(r"^(?:and\s+)?(?:what about|how about|tell me more|elaborate|continue|compare them)\b", question, re.I):
        return True
    if re.search(
        r"\b(?:first|second|third|fourth|last|previous)\s+(?:one|ones|thing|stage|step|process|question|answer)\b"
        r"|\b(?:you|we|i)\s+(?:mentioned|said|discussed|asked)\b"
        r"|\b(?:earlier|previously|there)\b", question, re.I,
    ):
        return True
    # These demonstratives name a previous conversational subject even when
    # the follow-up also names the other side of a comparison explicitly.
    if re.search(r"\b(?:this|that)\s+(?:location|process|step|stage|method|reaction)\b", question, re.I):
        return True
    reference = re.search(r"\b(?:it|its|they|their|them|those|these|former|latter)\b", question, re.I)
    if reference is None:
        return bool(re.match(r"^(?:why|how|what)\b.*\b(?:this|that)\s+(?:work|matter|important|mean|happen)", question, re.I))
    # An explicit named subject before a pronoun usually makes retrieval standalone.
    preceding = question[:reference.start()].split()[1:]
    return not any(re.fullmatch(r"[A-Z][A-Za-z0-9-]+", word.strip(".,?:")) for word in preceding)


def fallback_query(question, history, limit):
    previous = next((message.content for message in reversed(history) if isinstance(message, HumanMessage)), "")
    if not isinstance(previous, str) or not previous.strip():
        return question
    space = limit - len(question) - len("\nFollow-up: ")
    return f"{previous[:space]}\nFollow-up: {question}" if space >= 32 else question


def conservative_token_count(text):
    return len(text.encode("utf-8"))


def evidence_budget(question, query, history, settings, token_count=None):
    counter = token_count or conservative_token_count
    conversation = json.dumps(
        [{"role": message.type, "content": message.content} for message in history], ensure_ascii=False,
    )
    current = json.dumps({"question": question, "retrieval_query": query}, ensure_ascii=False)
    available = (
        settings.RAG_CONTEXT_WINDOW_TOKENS - settings.RAG_PROMPT_RESERVE_TOKENS
        - settings.RAG_OUTPUT_RESERVE_TOKENS - counter(conversation) - counter(current)
    )
    if available < 256:
        raise ChainInputError("The question and conversation exceed the configured context budget")
    return min(settings.RAG_MAX_EVIDENCE_TOKENS, available), counter


class CallMetrics(BaseCallbackHandler):
    def __init__(self):
        self.started = perf_counter()
        self.tokens = None

    def on_llm_end(self, response, **kwargs):
        usage = (response.llm_output or {}).get("token_usage")
        if not isinstance(usage, dict):
            for group in response.generations:
                for generation in group:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if isinstance(usage, dict):
                        break
        if isinstance(usage, dict):
            self.tokens = {
                key: value for key, value in usage.items()
                if key in {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
                and type(value) is int and value >= 0
            } or None

    def result(self, task, *, outcome="completed"):
        return {"task": task, "elapsed_ms": round((perf_counter() - self.started) * 1000, 2),
                "outcome": outcome, "tokens": self.tokens}
