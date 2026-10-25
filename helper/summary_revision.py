import re

_STRUCTURAL = re.compile(
    r"\b(?:reorganiz\w*|restructur\w*|outlines?|sections?|headings?|subsections?|"
    r"table of contents|main points|re-?order\w*|change the order|"
    r"merge(?: the)? sections|split(?: the)? sections?|"
    r"add(?: a| the)?(?: missing)? sections?|move(?: the)? sections?)\b",
    re.IGNORECASE,
)
_UNSUPPORTED = re.compile(
    r"\b(add|include|insert|invent|provide)\b.{0,40}\b("
    r"statistic|statistics|stats|percentage|case study|named example|"
    r"background|historical|not (in|from) (the )?(source|lecture|text)"
    r")\b",
    re.IGNORECASE,
)


def classify_summary_feedback(feedback):
    if not isinstance(feedback, str) or not feedback.strip():
        return "wording"
    text = feedback.strip()
    if text.casefold() in {"auto", "auto improve", "improve"}:
        return "wording"
    if _STRUCTURAL.search(text):
        return "structural"
    if _UNSUPPORTED.search(text):
        return "unsupported"
    return "wording"
