import hashlib
import json
from collections.abc import Mapping
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessageChunk, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from chains.contracts import text_output
from chains.schemes.QuestionSchem import Feedback


def _encoded(value):
    def supported(item):
        if isinstance(item, BaseModel):
            return item.model_dump()
        raise TypeError("History contains an unsupported value")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=supported)


def _digest(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _identifier(value, fallback):
    return value if isinstance(value, str) and 0 < len(value) <= 128 else fallback


def _legacy_messages(values):
    if not isinstance(values, (list, tuple)):
        return []
    result = []
    for position, value in enumerate(values):
        if isinstance(value, Mapping):
            role = value.get("type", value.get("role"))
            if not isinstance(role, str):
                continue
            cls = HumanMessage if role in {"human", "user"} else AIMessage if role in {"ai", "assistant"} else None
            if cls is None:
                continue
            try:
                value = cls(content=value["content"], id=value.get("id"))
            except (KeyError, TypeError, ValueError):
                continue
        if not isinstance(value, (HumanMessage, AIMessage)) or isinstance(value, BaseMessageChunk):
            continue
        try:
            content = text_output(value)
        except (TypeError, ValueError):
            continue
        identifier = value.id or "legacy-" + _digest([position, value.type, content])
        cls = HumanMessage if isinstance(value, HumanMessage) else AIMessage
        result.append(cls(content=content, id=identifier))
    return add_messages([], result)


def canonical_messages(state):
    messages = _legacy_messages(state.get("messages"))
    return messages or _legacy_messages(state.get("conversation_history"))


def merge_messages(left, right):
    # Stable legacy IDs allow checkpoint replacements and removal deltas.
    return add_messages(_legacy_messages(left), right)


def _turns(messages):
    turns = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns and len(turns[-1]) == 1:
            turns[-1].append(message)
    return [turn for turn in turns if len(turn) == 2]


def message_delta(state, answer, config):
    from chains.contracts import TextOutputError

    messages = canonical_messages(state)
    if not messages or not isinstance(messages[-1], HumanMessage):
        raise TextOutputError("An answer requires a pending human question")
    response = AIMessage(content=text_output(answer), id=str(uuid4()))
    turns = _turns(messages + [response])
    latest = turns[-1]
    max_messages = config.QA_RETENTION_MESSAGES
    max_characters = config.QA_RETENTION_CHARACTERS
    size = lambda turn: sum(len(message.content) for message in turn)
    if len(latest) > max_messages or size(latest) > max_characters:
        raise TextOutputError("The complete turn exceeds the configured history retention budget")
    selected = [latest]
    remaining_messages = max_messages - 2
    remaining_characters = max_characters - size(latest)
    if len(turns) > 1 and size(turns[0]) <= remaining_characters and remaining_messages >= 2:
        selected.insert(0, turns[0])
        remaining_messages -= 2
        remaining_characters -= size(turns[0])
    for turn in reversed(turns[1:-1]):
        if remaining_messages < 2 or size(turn) > remaining_characters:
            break
        selected.insert(1 if selected[0] is turns[0] else 0, turn)
        remaining_messages -= 2
        remaining_characters -= size(turn)
    keep = {message.id for turn in selected for message in turn}
    remove = [RemoveMessage(id=message.id) for message in messages if message.id not in keep]
    return {
        "messages": [*remove, response],
        "conversation_history": [],
        "history_trimmed": (state.get("history_trimmed", 0) if type(state.get("history_trimmed", 0)) is int else 0) + len(remove),
        "state_version": 4,
    }


def _legacy_critic(value):
    if not isinstance(value, str):
        return None
    try:
        return Feedback.model_validate_json(value)
    except (TypeError, ValueError):
        return None


def human_feedback_value(state):
    value = state.get("human_feedback_text")
    if isinstance(value, str):
        return value
    value = state.get("feedback", "")
    if not isinstance(value, str):
        return ""
    return "auto" if state.get("state_version") != 4 and _legacy_critic(value) is not None else value


def critic_feedback_value(state):
    value = state.get("critic_feedback")
    if isinstance(value, Feedback):
        return Feedback.model_validate(value.model_dump())
    if isinstance(value, Mapping):
        return Feedback.model_validate(dict(value))
    if state.get("state_version") != 4:
        return _legacy_critic(state.get("feedback"))
    return None


def _history(state):
    values = state.get("history", [])
    if not isinstance(values, list):
        return []
    result = {}
    for position, entry in enumerate(values):
        if not isinstance(entry, Mapping):
            continue
        try:
            _encoded(entry)
        except (TypeError, ValueError):
            continue
        feedback = entry.get("human_feedback", entry.get("feedback", ""))
        if not isinstance(feedback, str):
            continue
        critic = entry.get("critic_feedback")
        legacy = _legacy_critic(feedback)
        if legacy is not None:
            feedback, critic = "auto", legacy.model_dump()
        draft = entry.get("question", entry.get("draft_preview", ""))
        preview = draft.get("question", "") if isinstance(draft, Mapping) else draft if isinstance(draft, str) else ""
        preview = preview if isinstance(preview, str) else ""
        review_id = _identifier(entry.get("review_id") or entry.get("id"), "legacy-" + _digest([position, entry]))
        asset_id = _identifier(entry.get("asset_id", state.get("asset_id")), None)
        result.pop(review_id, None)
        result[review_id] = _compact_record({
            "id": review_id, "review_id": review_id,
            "asset_id": asset_id,
            "human_feedback": feedback,
            "critic_feedback": critic if isinstance(critic, Mapping) else None,
            "status": entry.get("status", "revised") if entry.get("status") in ("revised", "unchanged") else "revised",
            "reason": entry.get("reason") if isinstance(entry.get("reason"), str) else None,
            "draft_hash": _identifier(entry.get("draft_hash"), _digest(draft)), "draft_preview": preview[:256],
        })
    return list(result.values())


def _compact_record(record, budget=3072):
    record = dict(record)
    critic = record.get("critic_feedback")
    if isinstance(critic, Mapping) and "feedback_hash" not in critic:
        text = critic.get("feedback", "")
        record["critic_feedback"] = {
            "acceptable": critic.get("acceptable") is True,
            "issue_count": len(critic.get("issues", [])) if isinstance(critic.get("issues"), list) else 0,
            "feedback_hash": _digest(critic),
            "feedback_preview": text[:256] if isinstance(text, str) else "",
        }
    for key, limit in (("human_feedback", 2048), ("reason", 512), ("draft_preview", 256)):
        value = record.get(key)
        if isinstance(value, str):
            record[key] = value[:limit]
    while len(_encoded(record)) > budget:
        candidates = [key for key in ("human_feedback", "reason", "draft_preview") if isinstance(record.get(key), str) and len(record[key]) > 16]
        if candidates:
            key = max(candidates, key=lambda item: len(record[item]))
            record[key] = record[key][:len(record[key]) // 2]
        elif record.get("critic_feedback") is not None:
            record["critic_feedback"] = None
        else:
            from chains.contracts import ChainInputError
            raise ChainInputError("The revision history budget cannot hold a source reference")
    return record


def _requirements(state):
    values = state.get("revision_requirements", [])
    if not isinstance(values, list):
        values = []
    if not values:
        values = [{"review_id": entry["review_id"], "feedback": entry["human_feedback"]} for entry in _history(state)]
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("feedback"), str):
            continue
        feedback = value["feedback"].strip()
        key = feedback.casefold()
        if not key or key in {"auto", "save"} or key in seen:
            continue
        seen.add(key)
        requirement = {
            "review_id": _identifier(value.get("review_id"), "legacy-" + _digest(feedback)),
            "feedback": feedback,
        }
        if value.get("truncated") is True:
            requirement.update(truncated=True, feedback_hash=_identifier(value.get("feedback_hash"), _digest(feedback)))
        result.append(requirement)
    return result


def _bounded_requirements(values, budget):
    selected = []
    order = [len(values) - 1, 0, *range(len(values) - 2, 0, -1)] if len(values) > 1 else [0]
    for index in dict.fromkeys(order):
        if not values:
            break
        item = dict(values[index])
        item_budget = max(budget // 2, min(512, budget)) if not selected and len(values) > 1 else budget
        while len(_encoded([entry for _, entry in selected] + [item])) > item_budget:
            feedback = item["feedback"]
            if len(feedback) <= 32:
                break
            item.update(truncated=True, feedback_hash=item.get("feedback_hash", _digest(feedback)))
            length = len(feedback) // 4
            item["feedback"] = feedback[:length] + "\n[…]\n" + feedback[-length:]
        if len(_encoded([entry for _, entry in selected] + [item])) <= item_budget:
            selected.append((index, item))
    return [item for _, item in sorted(selected)]


def normalize_review_state(state):
    return {
        **state, "human_feedback_text": human_feedback_value(state),
        "critic_feedback": critic_feedback_value(state), "history": _history(state),
        "revision_requirements": _requirements(state),
    }


def revision_context(state, config):
    requirements = _bounded_requirements(_requirements(state), config.REVISION_CONTEXT_CHARACTERS // 2)
    records = _history(state)[-config.REVISION_CONTEXT_RECORDS:]
    context = ([{"retained_requirements": requirements}] if requirements else []) + records
    while len(context) > (1 if requirements else 0) and len(_encoded(context)) > config.REVISION_CONTEXT_CHARACTERS:
        context.pop(1 if requirements else 0)
    return context


def revision_delta(state, config, *, draft, status, reason=None, critic=None):
    if hasattr(draft, "model_dump"):
        draft = draft.model_dump()
    if not isinstance(draft, (Mapping, str)):
        from chains.contracts import StructuredOutputError
        raise StructuredOutputError("A revision draft must be a question or text")
    preview = draft.get("question", "") if isinstance(draft, Mapping) else draft
    review_id = _identifier(state.get("feedback_review_id"), "legacy-" + _digest([human_feedback_value(state), draft]))
    critic = critic or critic_feedback_value(state)
    if isinstance(critic, Feedback):
        critic = critic.model_dump()
    record = _compact_record({
        "id": review_id, "review_id": review_id,
        "asset_id": _identifier(state.get("asset_id"), None),
        "human_feedback": human_feedback_value(state), "critic_feedback": critic,
        "status": status, "reason": reason, "draft_hash": _digest(draft), "draft_preview": preview[:256],
    }, min(3072, config.REVISION_HISTORY_CHARACTERS - 2))
    history = [_compact_record(entry, min(3072, config.REVISION_HISTORY_CHARACTERS - 2)) for entry in _history(state) if entry["id"] != review_id] + [record]
    history = history[-config.REVISION_HISTORY_LIMIT:]
    while len(history) > 1 and len(_encoded(history)) > config.REVISION_HISTORY_CHARACTERS:
        history.pop(0)
    requirements = _requirements(state)
    feedback = human_feedback_value(state).strip()
    if feedback and feedback.casefold() not in {"auto", "save"}:
        requirements = [item for item in requirements if item["feedback"].casefold() != feedback.casefold()]
        requirements.append({"review_id": review_id, "feedback": feedback})
    requirements = _bounded_requirements(requirements, config.REVISION_REQUIREMENTS_CHARACTERS)
    return {"history": history, "revision_requirements": requirements, "state_version": 4}
