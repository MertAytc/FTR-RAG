"""Low-noise Turkish -> English clinical query processing for FTR RAG.

Design goals:
- Prefer the longest, most specific dictionary match.
- Match inflected Turkish phrases only as contiguous phrases (no bag-of-words matching).
- Suppress overlapping keys and duplicate/redundant English expansion terms.
- Keep the original query and preserve the Qwen rewrite API used by the project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import json
import re

try:
    import ollama
except ImportError:  # pragma: no cover - depends on deployment
    ollama = None

TERMINOLOGY_PATH = Path(__file__).with_name("clinical_terminology.json")
REWRITE_MODEL = "qwen3:8b"

# Conservative suffix list: only used to recover common Turkish inflections.
_SUFFIXES = tuple(sorted({
    "lerindeki", "larındaki", "lerimiz", "larımız", "leriniz", "larınız",
    "lerdir", "lardır", "lerinin", "larının", "lerine", "larına", "lerini", "larını",
    "lerden", "lardan", "lerle", "larla", "lerin", "ların", "ler", "lar",
    "daki", "deki", "taki", "teki", "dan", "den", "tan", "ten",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "nin", "nın", "nun", "nün", "nde", "nda", "yi", "yı", "yu", "yü",
    "ye", "ya", "ni", "nı", "nu", "nü", "ne", "na", "in", "ın", "un", "ün",
    "im", "ım", "um", "üm", "i", "ı", "u", "ü", "e", "a"
}, key=len, reverse=True))

# These terms are too generic to add to a biomedical BM25 query by themselves.
_GENERIC_ENGLISH = {
    "exercise", "progression", "progress", "movement", "activity", "function", "functional",
    "pain", "test", "scale", "assessment", "treatment", "therapy", "training", "load",
    "loading", "dose", "dosage", "condition", "disease", "risk", "symptom", "adult", "child",
    "age", "minutes", "seconds", "frequency", "intensity", "duration", "goal", "care", "evidence",
    "recommendation", "referral", "education", "safety", "monitoring", "examination", "evaluation",
    "history", "set", "sets", "repetition", "repetitions", "strength", "balance", "mobility", "walking",
    "gait", "shoulder", "knee", "hip", "hand", "arm", "leg", "neck", "back", "acute", "chronic",
    "mild", "moderate", "severe", "minutes", "seconds"
}

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.UNICODE)


def normalize(text: str) -> str:
    # Python's locale-independent lower() turns Turkish İ into i + combining dot.
    # Protect all-caps English abbreviations before Turkish-specific I handling:
    # FADI/ICF/FIM must normalize to fadi/icf/fim, not fadı/ıcf/fım.
    text = re.sub(r"\b[A-Z][A-Z0-9-]{1,}\b", lambda m: m.group(0).lower(), text)
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize(text))


def _stem_token(token: str) -> str:
    if len(token) <= 5:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix):
            candidate = token[:-len(suffix)]
            if len(candidate) >= 4:
                return candidate
    return token


def _term_token_sequences(term: str) -> List[str]:
    return [_stem_token(t) for t in _word_tokens(term)]


def _find_spans(query: str, term: str) -> List[Tuple[int, int]]:
    """Return token-index spans for exact or inflection-tolerant contiguous matches."""
    q_tokens = _word_tokens(query)
    t_tokens = _word_tokens(term)
    if not t_tokens or len(t_tokens) > len(q_tokens):
        return []

    # Exact token sequence first. Token-based matching prevents substring hits such as ROM in "from".
    spans: List[Tuple[int, int]] = []
    n = len(t_tokens)
    for i in range(len(q_tokens) - n + 1):
        window = q_tokens[i:i+n]
        if window == t_tokens:
            spans.append((i, i+n))
    if spans:
        return spans

    # Conservative morphological match: all tokens must match in order and contiguously.
    q_stems = [_stem_token(t) for t in q_tokens]
    t_stems = [_stem_token(t) for t in t_tokens]
    for i in range(len(q_stems) - n + 1):
        if q_stems[i:i+n] == t_stems:
            spans.append((i, i+n))
    return spans


def _phrase_matches(query: str, term: str) -> bool:
    return bool(_find_spans(query, term))


def load_terminology(path: Path = TERMINOLOGY_PATH) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("clinical_terminology.json düz JSON nesnesi olmalı.")
    for key, values in data.items():
        if not isinstance(key, str) or not isinstance(values, list) or not all(isinstance(x, str) for x in values):
            raise ValueError(f"Geçersiz sözlük girdisi: {key!r}")
    return data


def find_terms(query: str, terminology: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """Match most-specific non-overlapping Turkish terms, preserving query order."""
    candidates = []
    for key, values in terminology.items():
        for start, end in _find_spans(query, key):
            candidates.append((start, end, key, values))
    # Longest span wins at the same start; then longer character key wins.
    candidates.sort(key=lambda x: (x[0], -(x[1]-x[0]), -len(x[2])))
    selected = []
    occupied = set()
    seen_keys = set()
    for start, end, key, values in candidates:
        if key in seen_keys or any(i in occupied for i in range(start, end)):
            continue
        selected.append((start, key, values))
        occupied.update(range(start, end))
        seen_keys.add(key)
    selected.sort(key=lambda x: x[0])
    return [(key, values) for _, key, values in selected]


def _phrase_norm(text: str) -> str:
    return " ".join(_word_tokens(text))


def _unique_phrases(items: Iterable[str]) -> List[str]:
    """Deduplicate case/punctuation variants and discard generic or contained synonyms."""
    candidates: List[str] = []
    seen = set()
    for item in items:
        item = re.sub(r"\s+", " ", item.strip())
        norm = _phrase_norm(item)
        if not norm or norm in seen:
            continue
        tokens = norm.split()
        if len(tokens) == 1 and norm in _GENERIC_ENGLISH:
            continue
        seen.add(norm)
        candidates.append(item)

    # If a phrase is merely a fragment of another phrase in this same expansion set,
    # discard the fragment (e.g., "progression" beside "exercise progression").
    kept = []
    norms = [_phrase_norm(x) for x in candidates]
    for idx, phrase in enumerate(candidates):
        norm = norms[idx]
        tokens = norm.split()
        if len(tokens) == 1 and any(norm in other.split() for j, other in enumerate(norms) if j != idx and len(other.split()) > 1):
            continue
        kept.append(phrase)
    return kept


def _contains_term(text: str, term: str) -> bool:
    norm_text = _phrase_norm(text)
    norm_term = _phrase_norm(term)
    if not norm_text or not norm_term:
        return False
    return f" {norm_term} " in f" {norm_text} "


def _matched_expansions(matched_terms: List[Tuple[str, List[str]]]) -> List[str]:
    return _unique_phrases(value for _, values in matched_terms for value in values)


def _preserve_terminology_anchors(rewritten: str, matched_terms: List[Tuple[str, List[str]]]) -> str:
    """Add only one precise anchor per matched concept when rewrite omitted it."""
    additions = []
    for _, values in matched_terms:
        choices = _unique_phrases(values)
        if not choices:
            continue
        if not any(_contains_term(rewritten, candidate) for candidate in choices):
            additions.append(choices[0])
    anchors = _unique_phrases(additions)
    if not anchors:
        return rewritten.strip()
    return f"{rewritten.strip()} {' '.join(anchors)}".strip()


def rewrite_query_to_english(query: str, matched_terms: List[Tuple[str, List[str]]]) -> str:
    if ollama is None:
        raise RuntimeError("ollama paketi bulunamadı. English rewrite için Ollama gerekli.")

    hints = _matched_expansions(matched_terms)
    hint_text = "; ".join(hints[:12]) if hints else "none"
    prompt = f"""Rewrite the following Turkish physiotherapy/rehabilitation question into precise clinical English for information retrieval.

Rules:
- Preserve the exact clinical intent and all condition, population, intervention, outcome, progression, dosage, safety, and comparison details.
- Preserve explicit numbers, units, time periods, and abbreviations exactly.
- Prefer standard terms used in English physiotherapy clinical practice guidelines.
- Do not answer the question and do not add facts not present in the Turkish question.
- Use the terminology hints only when they match the intended concept; do not append a list of synonyms.
- Return only the rewritten English question, without explanation or quotation marks.

Turkish question:
{query}

Relevant terminology hints:
{hint_text}
"""
    response = ollama.chat(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": "You rewrite clinical queries only. Preserve intent, never answer, and return only the English query."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0, "num_predict": 120},
        think=False,
    )
    rewritten = response["message"]["content"].strip().strip('"')
    if not rewritten:
        raise RuntimeError("English query rewrite boş döndü.")
    return _preserve_terminology_anchors(rewritten, matched_terms)


def _dedupe_expansion_tokens(base_query: str, phrases: List[str]) -> str:
    """BM25 is bag-of-words: append each informative expansion token only once."""
    output_tokens = []
    seen = set(_word_tokens(base_query))
    for phrase in phrases:
        phrase_tokens = _word_tokens(phrase)
        # Single-token generic phrases are already removed by _unique_phrases.
        # Within a clinically meaningful phrase, retain useful tokens such as
        # "exercise" or "test", but emit each token only once across all expansions.
        for token in phrase_tokens:
            if token in seen:
                continue
            if len(token) == 1 and not token.isdigit():
                continue
            seen.add(token)
            output_tokens.append(token)
    if not output_tokens:
        return base_query.strip()
    return f"{base_query.strip()} {' '.join(output_tokens)}".strip()


def expand_query(
    query: str,
    terminology: Dict[str, List[str]] | None = None,
    matched_terms: List[Tuple[str, List[str]]] | None = None,
) -> str:
    if terminology is None:
        terminology = load_terminology()
    if matched_terms is None:
        matched_terms = find_terms(query, terminology)
    phrases = _matched_expansions(matched_terms)
    return _dedupe_expansion_tokens(query.strip(), phrases)


def process_query(
    query: str,
    terminology: Dict[str, List[str]] | None = None,
    use_english_rewrite: bool = True,
) -> Dict[str, object]:
    if not query or not query.strip():
        raise ValueError("Sorgu boş olamaz.")
    if terminology is None:
        terminology = load_terminology()

    original_query = query.strip()
    matches = find_terms(original_query, terminology)
    added_terms = _matched_expansions(matches)
    rewritten_query = rewrite_query_to_english(original_query, matches) if use_english_rewrite else original_query
    expanded_query = _dedupe_expansion_tokens(rewritten_query, added_terms)

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "expanded_query": expanded_query,
        "matched_terms": [term for term, _ in matches],
        "added_terms": added_terms,
    }
