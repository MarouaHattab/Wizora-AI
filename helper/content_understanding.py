import hashlib
import json
import math
import re
import time
import unicodedata
from collections import OrderedDict
from copy import deepcopy
from difflib import SequenceMatcher
from threading import BoundedSemaphore, Event, RLock

_EXAMPLE = re.compile(
    r"^(?:(?:worked|numerical|illustrative)\s+)?example\b|^demonstration\b|^exemple(?:\s+(?:résolu|corrigé|numérique))?\b|^مثال(?:\s+محلول)?",
    re.I,
)
_SAMPLE = re.compile(
    r"^(?:(?:sample|practice|review|exam)\s+)?question\b|^q\s*\d+[.:)]|^question\s+(?:type|exemple)|^(?:سؤال\s+(?:نموذجي|تدريبي)|نموذج\s+سؤال)",
    re.I,
)
_EXERCISE = re.compile(
    r"^(?:(?:practice|review|exam)\s+)?(?:exercise|problem)\b|^exercice\b|^problème\b|^(?:تمرين|مسألة)\b",
    re.I,
)
_SOLUTION = re.compile(
    r"^(?:solution|answer|calculation|working|réponse|calcul|corrigé)\s*[:.]|^(?:الحل|الإجابة|الجواب|الحساب)\s*[:：]",
    re.I,
)
_DEFINITION = re.compile(
    r"^(?:definition|définition|تعريف)\s*[:：]|\b(?:is defined as|est défini(?:e)? comme|on appelle)\b|(?:يُعرّف|تعرف بأنها|يعرف بأنه)",
    re.I,
)
_EXPLANATION = re.compile(
    r"^(?:theory|concept|definition|principle|explanation|discussion|overview|conclusion|théorie|définition|principe|explication|تعريف|شرح|مفهوم|نظرية)\s*[:.：]",
    re.I,
)
_PREFIX = re.compile(
    r"^(?:(?:worked|numerical|illustrative|sample|practice|review|exam)\s+)?(?:example|question|exercise|problem|demonstration)(?:\s+\d+)?\s*[:.)-]?\s*|^(?:exemple(?:\s+(?:résolu|corrigé|numérique))?|exercice|problème|question(?:\s+type)?|مثال(?:\s+محلول)?|تمرين|مسألة|سؤال(?:\s+نموذجي)?)(?:\s+\d+)?\s*[:.)：-]?\s*",
    re.I,
)
_IMPERATIVE = re.compile(
    r"^(?:calculate|determine|find|solve|explain why|compare|calculez|calculer|déterminez|déterminer|trouvez|résoudre|expliquez|احسب|حدد|حل|قارن|فسر)\b",
    re.I,
)
_STEPS = re.compile(
    r"(?:\b(?:first|then|therefore|the result|we obtain|divide|substitute|d'abord|ensuite|donc|on obtient|on divise|résultat)\b|أولا|ثم|إذن|النتيجة|نقسم)",
    re.I,
)
_ROLES = {
    "explanation",
    "definition",
    "worked_example",
    "existing_exercise",
    "sample_question",
    "visual_example",
}
_SEMANTIC_CACHE = OrderedDict()
_SEMANTIC_INFLIGHT = {}
_SEMANTIC_LOCK = RLock()
_SEMANTIC_WORKERS = BoundedSemaphore(2)
_SEMANTIC_MAX_BLOCKS = 16
_SEMANTIC_MAX_BYTES = 9000
_PHRASE_ARTICLES = {
    "the",
    "a",
    "an",
    "of",
    "le",
    "la",
    "les",
    "un",
    "une",
    "du",
    "de",
    "des",
    "d",
    "l",
}
_QUESTION_FORMS = {
    "doubling": "double",
    "doubled": "double",
    "doubles": "double",
    "doublement": "double",
    "doublee": "double",
    "doubler": "double",
    "updates": "update",
    "parameters": "parameter",
    "gradients": "gradient",
    "magnitudes": "magnitude",
    "controls": "control",
    "controlled": "control",
    "controlling": "control",
    "determines": "determine",
    "affects": "affect",
}
_QUESTION_NEGATION = {
    "not",
    "no",
    "never",
    "cannot",
    "without",
    "pas",
    "jamais",
    "sans",
    "non",
    "لا",
    "ليس",
    "ليست",
    "غير",
    "بدون",
}
_QUESTION_CONDITIONS = {
    "zero",
    "nonzero",
    "positive",
    "negative",
    "nul",
    "nulle",
    "positif",
    "negatif",
    "صفر",
    "موجب",
    "سالب",
}


def _math_marks(text):
    # NFC retains exponent/subscript identity; NFKC would collapse x² and x₂.
    return tuple(
        char
        for char in unicodedata.normalize("NFC", text)
        if unicodedata.category(char) == "Sm"
        or char in "+-*/=^<>_%|·"
        or "ARROW" in unicodedata.name(char, "")
        or "SUPERSCRIPT" in unicodedata.name(char, "")
        or "SUBSCRIPT" in unicodedata.name(char, "")
        or "MATHEMATICAL" in unicodedata.name(char, "")
        or unicodedata.decomposition(char).startswith("<font>")
    )


def _math_operands(text):
    text = unicodedata.normalize("NFC", text)
    marks, result = set(_math_marks(text)), []
    for index, char in enumerate(text):
        if char not in marks:
            continue
        before, after = re.findall(r"\w+", text[:index]), re.findall(r"\w+", text[index + 1 :])
        unary = "SUPERSCRIPT" in unicodedata.name(char, "") or "SUBSCRIPT" in unicodedata.name(
            char, ""
        )
        # Case can identify different variables (A versus a), even in prose.
        result.append(
            (char, before[-1] if before else "", "" if unary else after[0] if after else "")
        )
    return tuple(result)


def _question_words(text):
    # Accent folding is useful for language matching; compatibility folding can
    # erase mathematical alphabets, so retain those distinctions.
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text).casefold()
        if not unicodedata.combining(char)
    )
    text = re.sub(r"n['’]t\b", " not", text)
    return [_QUESTION_FORMS.get(token, token) for token in re.findall(r"[^\W_]+", text, re.UNICODE)]


def _effect_roles(words):
    patterns = (
        ("en", r"how does (.+?) affect (.+)", False),
        ("en", r"what effect does (.+?) have on (.+)", False),
        ("fr", r"quel effet a (.+?) sur (.+)", False),
        ("fr", r"comment (.+?) change avec (.+)", True),
    )
    for language, pattern, reverse in patterns:
        match = re.fullmatch(pattern, " ".join(words))
        if match is None:
            continue
        subject, outcome = match.groups()[::-1] if reverse else match.groups()

        def phrase(value, *, is_subject=False):
            tokens = value.split()
            while tokens and tokens[0] in _PHRASE_ARTICLES:
                tokens.pop(0)
            # "magnitude of X" and "X magnitude" share this precise noun role.
            conditions = {
                "with",
                "when",
                "if",
                "before",
                "after",
                "without",
                "under",
                "at",
                "during",
                "from",
                "to",
                "in",
                "on",
                "by",
                "while",
                "or",
                "and",
            }
            if tokens[:2] == ["magnitude", "of"] and not conditions.intersection(tokens[2:]):
                tokens = tokens[2:] + ["magnitude"]
            tokens = [token for token in tokens if token not in _PHRASE_ARTICLES]
            # French places the same doubling modifier before or after its noun.
            if is_subject and language == "fr" and tokens and tokens[-1] == "double":
                tokens = ["double"] + tokens[:-1]
            return tuple(tokens)

        roles = phrase(subject, is_subject=True), phrase(outcome)
        return (language, *roles) if all(roles) else None
    return None


def question_near_duplicate(left, right):
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if _math_marks(left) != _math_marks(right) or _math_operands(left) != _math_operands(right):
        return False

    def normalized(value):
        return unicodedata.normalize("NFKC", value).casefold()

    if re.findall(r"\d+(?:[.,]\d+)?", normalized(left)) != re.findall(
        r"\d+(?:[.,]\d+)?", normalized(right)
    ):
        return False
    one, two = _question_words(left), _question_words(right)
    for protected in (_QUESTION_NEGATION, _QUESTION_CONDITIONS):
        if set(one) & protected != set(two) & protected:
            return False
    if one == two and bool(one):
        return True

    def possibility(words):
        text = " ".join(words)
        verbs = r"(?:reduce|increase|decrease|change|update|use|calculate|compute|control|produce)"
        direct = re.fullmatch(r"can (.+?) (" + verbs + r" .+)", text)
        indirect = re.fullmatch(
            r"is it possible to ("
            + verbs
            + r" .+?) in (.+?)( (?:without|after|before|when|if) .+)?",
            text,
        )
        if direct:
            subject, action = direct.groups()
        elif indirect:
            action, subject, condition = indirect.groups()
            action += condition or ""
        else:
            return None

        def clean(value):
            return tuple(
                token for token in value.split() if token not in _PHRASE_ARTICLES | {"its"}
            )

        return clean(subject), clean(action)

    first_possibility, second_possibility = possibility(one), possibility(two)
    if first_possibility is not None and first_possibility == second_possibility:
        return True
    first, second = _effect_roles(one), _effect_roles(two)
    return first is not None and first == second


def _tokens(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return re.findall(r"[^\W\d_]+|#", re.sub(r"\d+(?:[.,]\d+)?", "#", text), re.UNICODE)


def _near_copy(left, right):
    if _math_marks(left) != _math_marks(right):
        return False
    a, b = _tokens(_PREFIX.sub("", left)), _tokens(_PREFIX.sub("", right))
    if min(len(a), len(b)) < 5:
        return a == b and bool(a)
    if a == b or SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.78:
        return True

    def shingles(tokens):
        return {tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)}

    one, two = shingles(a), shingles(b)
    return len(one & two) >= 2 and len(one & two) / max(1, min(len(one), len(two))) >= 0.72


def _source_blocks(context):
    blocks, lines, locator = [], [], {}

    def flush():
        if lines and len(blocks) < 256:
            blocks.append(
                {
                    "block_id": f"block-{len(blocks) + 1}",
                    "text": "\n".join(lines)[:6000],
                    "source_locator": dict(locator),
                }
            )
        lines.clear()

    for raw in context.splitlines():
        line = raw.strip()
        if line.startswith("SOURCE_LOCATOR: "):
            flush()
            try:
                candidate = json.loads(line.removeprefix("SOURCE_LOCATOR: "))
                locator = candidate if isinstance(candidate, dict) else {}
            except ValueError:
                locator = {}
        elif not line:
            flush()
        elif line not in {"SOURCE_TEXT:", "<SOURCE>", "</SOURCE>"}:
            if lines and any(
                pattern.search(line) for pattern in (_EXAMPLE, _SAMPLE, _EXERCISE, _EXPLANATION)
            ):
                flush()
            lines.append(line)
    flush()
    return blocks


def _classify_block(block, previous=None, following=None):
    text, locator = block["text"], block["source_locator"]
    visual = " ".join(
        str(locator.get(key, ""))
        for key in ("content_type", "record_type", "source_type", "type", "extraction_method")
    )
    if re.search(r"visual|image|diagram|figure|chart", visual, re.I) or re.search(
        r"(?:flowchart|diagram|figure|graph|schéma|diagramme|مخطط|رسم).{0,35}(?:illustrates|shows|depicts|montre|illustre|يوضح|يبين)",
        text,
        re.I,
    ):
        return "visual_example", 0.8, "visual locator or explanatory visual relationship"
    if _SAMPLE.search(text):
        return "sample_question", 0.9, "assessment heading and question context"
    if _EXERCISE.search(text) or _IMPERATIVE.search(text):
        return "existing_exercise", 0.85, "instruction requests an answer or calculation"
    quantities = re.findall(r"\d+(?:[.,]\d+)?", text)
    derived_result = re.search(
        r"(?:\b(?:giving|yielding|resulting in|gives|donne|donnant|soit|ce qui donne)\b|يعطي|بالتالي|فتكون).{0,100}\d",
        text,
        re.I,
    )
    if (
        _EXAMPLE.search(text)
        or (_STEPS.search(text) and (quantities or _SOLUTION.search(text)))
        or (len(quantities) >= 3 and derived_result)
    ):
        return "worked_example", 0.85, "worked scenario with solution steps or result"
    if _SOLUTION.search(text):
        prior = previous.get("classification") if previous else None
        return (
            (prior if prior in {"sample_question", "existing_exercise"} else "worked_example"),
            0.75,
            "solution continues the neighboring educational block",
        )
    if re.search(r"\d", text) and following and _SOLUTION.search(following["text"]):
        return "worked_example", 0.75, "scenario followed by a solution"
    if _DEFINITION.search(text):
        return "definition", 0.85, "definition structure"
    if re.search(r"[?؟]\s*$", text):
        return "sample_question", 0.65, "unlabelled interrogative assessment block"
    return (
        "explanation",
        0.55,
        "conservative narrative fallback; no exercise or solution structure detected",
    )


def understand_content(context, *, semantic_classifier=None):
    if not isinstance(context, str):
        raise ValueError("Content understanding requires extracted text.")
    result = {
        "schema_version": "1.1",
        "method": "multilingual_contextual_fallback",
        "explanatory_content": [],
        "definitions": [],
        "worked_examples": [],
        "existing_exercises": [],
        "sample_questions": [],
        "visual_examples": [],
        "concepts": [],
        "classification_warnings": [],
    }
    blocks = _source_blocks(context)
    for index, block in enumerate(blocks):
        role, confidence, basis = _classify_block(
            block,
            blocks[index - 1] if index else None,
            blocks[index + 1] if index + 1 < len(blocks) else None,
        )
        block.update(classification=role, confidence=confidence, classification_basis=basis)
    if semantic_classifier is not None:
        try:
            projected = [
                {
                    **block,
                    "preceding_text": blocks[index - 1]["text"][:800] if index else "",
                    "following_text": blocks[index + 1]["text"][:800]
                    if index + 1 < len(blocks)
                    else "",
                }
                for index, block in enumerate(blocks)
            ]
            decisions = semantic_classifier(json.loads(json.dumps(projected, ensure_ascii=False)))
            by_id = {block["block_id"]: block for block in blocks}
            if not isinstance(decisions, list):
                raise ValueError("Classification must return a list")
            for decision in decisions:
                if (
                    not isinstance(decision, dict)
                    or decision.get("block_id") not in by_id
                    or decision.get("classification") not in _ROLES
                ):
                    raise ValueError("Unknown educational block or role")
                confidence = decision.get("confidence", 0.7)
                if (
                    type(confidence) not in {int, float}
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                ):
                    raise ValueError("Invalid confidence")
            for decision in decisions:
                by_id[decision["block_id"]].update(
                    classification=decision["classification"],
                    confidence=decision.get("confidence", 0.7),
                    classification_basis="explicit semantic classifier hook",
                )
            result["method"] = "semantic_hook_with_multilingual_contextual_fallback"
        except Exception:
            result["classification_warnings"].append(
                "The optional semantic classifier was unavailable or invalid; conservative multilingual local classification was retained."
            )
    result["blocks"] = blocks
    keys = {
        "explanation": "explanatory_content",
        "definition": "definitions",
        "worked_example": "worked_examples",
        "existing_exercise": "existing_exercises",
        "sample_question": "sample_questions",
        "visual_example": "visual_examples",
    }
    for block in blocks:
        key = keys[block["classification"]]
        if len(result[key]) < 64:
            result[key].append(block)
        if key == "definitions":
            result["explanatory_content"].append(block)
    concepts, seen = [], set()
    # Explanatory evidence is preferred. When a lesson consists of worked examples,
    # explicit solution explanations may still identify a concept without serving as stems.
    evidence = (
        result["explanatory_content"] + result["visual_examples"] or result["worked_examples"]
    )
    for block in evidence:
        for sentence in re.split(r"(?<=[.!?])\s+|;\s*", block["text"]):
            sentence = _EXPLANATION.sub("", sentence).strip()
            key = " ".join(_tokens(sentence))
            if (
                len(_tokens(sentence)) < 5
                or key in seen
                or _SAMPLE.search(sentence)
                or _EXERCISE.search(sentence)
            ):
                continue
            seen.add(key)
            concepts.append(
                {
                    "concept_id": f"concept-{len(concepts) + 1}",
                    "title": " ".join(sentence.split()[:9]).rstrip(".!?"),
                    "evidence": sentence[:700],
                    "source_locator": block["source_locator"],
                }
            )
            if len(concepts) >= 24:
                break
        if len(concepts) >= 24:
            break
    result["concepts"] = concepts
    result["coverage_note"] = (
        "Concept candidates come from extracted source evidence. Multilingual block roles are contextual estimates; uncertain roles require review. Worked examples, existing exercises and sample questions support understanding and must not be copied as assessment stems. Local classification is bounded to 256 blocks of 6000 characters; the full source remains unchanged."
    )
    return result


def clear_content_classification_cache():
    with _SEMANTIC_LOCK:
        _SEMANTIC_CACHE.clear()


def _classification_setting(settings, name, default=None):
    return (
        settings.get(name, default)
        if isinstance(settings, dict)
        else getattr(settings, name, default)
    )


def _classification_key(context, settings):
    names = (
        "QUESTION_GENERATER_PROVIDER",
        "QUESTION_GENERATER_MODEL_ID",
        "QUESTION_GENERATER_TEMPERATURE",
        "OLLAMA_BASE_URL",
        "OLLAMA_NUM_CTX",
        "PROVIDER_FALLBACK_ENABLED",
        "PROVIDER_FALLBACK_ALLOW_PAID",
        "PROVIDER_FALLBACK_PROVIDER",
        "PROVIDER_FALLBACK_MODEL_ID",
    )
    model = tuple(str(_classification_setting(settings, name, "")) for name in names)
    return ("educational-roles-v1", hashlib.sha256(context.encode()).hexdigest(), model)


def _semantic_projection(understanding):
    blocks, selected, measured = understanding["blocks"], [], 0
    for index, block in enumerate(blocks):
        if block["confidence"] >= 0.86:
            continue
        projection = {
            "block_id": block["block_id"],
            "text": block["text"],
            "source_locator": block["source_locator"],
            "preceding_text": blocks[index - 1]["text"][:300] if index else "",
            "following_text": blocks[index + 1]["text"][:300] if index + 1 < len(blocks) else "",
        }
        size = len(json.dumps(projection, ensure_ascii=False).encode())
        if len(selected) < _SEMANTIC_MAX_BLOCKS and measured + size <= _SEMANTIC_MAX_BYTES:
            selected.append(projection)
            measured += size
    return selected


def configured_understanding(context, settings, *, chain_factory=None, config=None):
    local = understand_content(context)
    metadata = {
        "status": "disabled",
        "cache_hit": False,
        "model_calls": 0,
        "provider": _classification_setting(settings, "QUESTION_GENERATER_PROVIDER"),
        "model": _classification_setting(settings, "QUESTION_GENERATER_MODEL_ID"),
        "max_blocks": _SEMANTIC_MAX_BLOCKS,
        "max_input_bytes": _SEMANTIC_MAX_BYTES,
        "blocks_classified": 0,
        "source_block_count": len(local["blocks"]),
    }
    local["semantic_classification"] = metadata
    if not _classification_setting(settings, "SEMANTIC_CONTENT_CLASSIFICATION_ENABLED", True):
        return local
    key = _classification_key(context, settings)
    with _SEMANTIC_LOCK:
        cached = _SEMANTIC_CACHE.get(key)
        if cached is not None and cached[0] > time.monotonic():
            _SEMANTIC_CACHE.move_to_end(key)
            result = deepcopy(cached[1])
            result["semantic_classification"].update(cache_hit=True, model_calls=0)
            return result
        event = _SEMANTIC_INFLIGHT.get(key)
        owner = event is None
        if owner:
            event = _SEMANTIC_INFLIGHT[key] = Event()
    if not owner:
        event.wait(timeout=float(_classification_setting(settings, "LLM_TIMEOUT_SECONDS", 60)) + 5)
        with _SEMANTIC_LOCK:
            cached = _SEMANTIC_CACHE.get(key)
            if cached is not None and cached[0] > time.monotonic():
                result = deepcopy(cached[1])
                result["semantic_classification"].update(cache_hit=True, model_calls=0)
                return result
        metadata["status"] = "fallback_busy"
        local["classification_warnings"].append(
            "Semantic classification is still running for this source; conservative local roles were retained."
        )
        return local
    result = local
    try:
        projection = _semantic_projection(local)
        if not projection:
            metadata["status"] = (
                "local_confident"
                if all(block["confidence"] >= 0.86 for block in local["blocks"])
                else "fallback_budget"
            )
            if metadata["status"] == "fallback_budget":
                local["classification_warnings"].append(
                    "Source blocks exceed the bounded semantic input budget; local roles were retained."
                )
        else:
            from chains.ContentClassificationChain import ContentClassificationResult

            if chain_factory is None:
                from chains import get_task_chain

                chain_factory = get_task_chain
            acquired = _SEMANTIC_WORKERS.acquire(
                timeout=float(_classification_setting(settings, "LLM_TIMEOUT_SECONDS", 60))
            )
            if not acquired:
                raise TimeoutError("Semantic classifier worker budget exhausted")
            try:
                metadata["model_calls"] = 1
                response = chain_factory("content_classification", settings).invoke(
                    {"blocks": projection}, config=config
                )
            finally:
                _SEMANTIC_WORKERS.release()
            response = ContentClassificationResult.model_validate(
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            if sorted(item.block_id for item in response.decisions) != sorted(
                block["block_id"] for block in projection
            ):
                raise ValueError("Semantic classification changed source block identities")
            decisions = [item.model_dump() for item in response.decisions]
            result = understand_content(context, semantic_classifier=lambda blocks: decisions)
            metadata.update(status="model_assessed", blocks_classified=len(decisions))
            if any(item.confidence < 0.65 for item in response.decisions):
                result["classification_warnings"].append(
                    "Some semantic educational roles have low model confidence and require review."
                )
            ambiguous_count = sum(block["confidence"] < 0.86 for block in local["blocks"])
            if len(decisions) < ambiguous_count:
                result["classification_warnings"].append(
                    "The semantic block/input budget was reached; remaining blocks use explicit local fallback roles."
                )
    except Exception:
        metadata["status"] = "fallback_unavailable"
        local["classification_warnings"].append(
            "The configured semantic classifier was unavailable or returned invalid roles; conservative multilingual local classification was retained."
        )
        result = local
    finally:
        result["semantic_classification"] = dict(metadata)
        ttl = 300 if metadata["status"] in {"model_assessed", "local_confident"} else 15
        with _SEMANTIC_LOCK:
            _SEMANTIC_CACHE[key] = (time.monotonic() + ttl, deepcopy(result))
            _SEMANTIC_CACHE.move_to_end(key)
            while len(_SEMANTIC_CACHE) > 16:
                _SEMANTIC_CACHE.popitem(last=False)
            _SEMANTIC_INFLIGHT.pop(key, None)
            event.set()
    return result


def project_understanding(context, classified):
    def identity(block):
        return block["text"], json.dumps(
            block["source_locator"], sort_keys=True, ensure_ascii=False
        )

    roles = {identity(block): block for block in classified.get("blocks", [])}

    def decisions(blocks):
        return [
            {
                "block_id": block["block_id"],
                "classification": roles[identity(block)]["classification"],
                "confidence": roles[identity(block)]["confidence"],
            }
            for block in blocks
            if identity(block) in roles
        ]

    result = understand_content(context, semantic_classifier=decisions)
    result["method"] = classified.get("method", result["method"])
    result["classification_warnings"] = list(classified.get("classification_warnings", []))
    if "semantic_classification" in classified:
        result["semantic_classification"] = dict(classified["semantic_classification"])
    return result


def understanding_prompt(value):
    return json.dumps(
        {
            "concepts": [
                {"concept_id": item["concept_id"], "evidence": item["evidence"][:280]}
                for item in value.get("concepts", [])[:8]
            ],
            "source_examples_are_support_only": True,
            "classification_method": value.get("method"),
            "avoid_copying": [
                block["text"][:220]
                for key in ("sample_questions", "worked_examples", "existing_exercises")
                for block in value.get(key, [])[:3]
            ],
        },
        ensure_ascii=False,
    )


def originality_issue(stem, understanding, previous=()):
    for question in previous:
        if question_near_duplicate(stem, question):
            return "Generate an original question about another supported concept; this stem repeats or closely paraphrases a previous question."
    for key in ("sample_questions", "worked_examples", "existing_exercises"):
        for block in understanding.get(key, []):
            text = re.split(
                r"\b(?:solution|answer|réponse|الحل|الإجابة)\s*[:：]",
                block["text"],
                maxsplit=1,
                flags=re.I,
            )[0]
            if _near_copy(stem, text):
                return "Generate an original question from the underlying educational concept. Do not copy a worked example/sample question or merely change its numbers or names."
    return None


def grounding_issue(question, understanding):
    if question.question_type != "T/F":
        return None
    facts = {}
    pattern = re.compile(r"\b([a-z][a-z ]{0,30}?)\s+(?:is\s+)?measured\s+in\s+([a-z]+)\b", re.I)
    for block in understanding.get("explanatory_content", []):
        for subject, unit in pattern.findall(block["text"]):
            subject = re.sub(
                r"^(?:theory|definition|concept)\s*:\s*", "", subject.strip(), flags=re.I
            )
            facts[subject.casefold()] = unit.casefold()
    for subject, unit in pattern.findall(question.question):
        expected = facts.get(subject.strip().casefold())
        if expected is not None and (unit.casefold() == expected) != (question.answer == "True"):
            return "The True/False answer contradicts an explicit source definition or measurement unit. Check the exact quantity and units; do not interchange related concepts."
    return None
