from pathlib import Path
from typing import Dict, List, Tuple
import json
import re

try:
    import ollama
except ImportError:
    ollama = None

TERMINOLOGY_PATH = Path(__file__).with_name("clinical_terminology.json")
REWRITE_MODEL = "qwen3:8b"

_SUFFIXES = (
    "lerindeki", "larındaki", "lerimiz", "larımız", "leriniz", "larınız",
    "lerdir", "lardır", "lerinin", "larının", "lerine", "larına", "lerini", "larını",
    "lerden", "lardan", "lerle", "larla", "lerin", "ların", "ler", "lar",
    "daki", "deki", "taki", "teki", "dan", "den", "tan", "ten",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "nin", "nın", "nun", "nün", "nde", "nda", "yi", "yı", "yu", "yü",
    "ye", "ya", "ni", "nı", "nu", "nü", "ne", "na", "in", "ın", "un", "ün",
    "im", "ım", "um", "üm", "i", "ı", "u", "ü", "e", "a"
)


def load_terminology(path: Path = TERMINOLOGY_PATH) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    # Türkçe İ/I -> Python'un locale-bağımsız str.lower()'ı "İ"yi 'i' + U+0307
    # (combining dot above) olarak iki karaktere çeviriyor; bu da büyük harfle
    # başlayan Türkçe terimlerin (İnme, İlaç, ...) tokenization'da bölünüp
    # terminoloji sözlüğüyle hiç eşleşmemesine yol açıyordu. Önce Türkçe
    # büyük/küçük harf eşlemesini elle yapıp sonra lower() çağırıyoruz.
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zçğıöşü0-9]+", normalize(text), flags=re.UNICODE)


def _stem_token(token: str) -> str:
    if len(token) <= 5:
        return token
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if token.endswith(suffix):
            candidate = token[:-len(suffix)]
            if len(candidate) >= 4:
                return candidate
    return token


def _phrase_matches(query: str, term: str) -> bool:
    q = normalize(query)
    t = normalize(term)

    if t:
        escaped = re.escape(t)
        if re.search(rf"(?<![a-zçğıöşü0-9]){escaped}(?![a-zçğıöşü0-9])", q):
            return True

    q_stems = [_stem_token(token) for token in _word_tokens(q)]
    t_stems = [_stem_token(token) for token in _word_tokens(t)]
    if not t_stems:
        return False
    if len(t_stems) == 1:
        return t_stems[0] in q_stems

    for i in range(len(q_stems) - len(t_stems) + 1):
        if q_stems[i:i + len(t_stems)] == t_stems:
            return True

    if len(t_stems) == 2 and all(stem in q_stems for stem in t_stems):
        return True
    return False


def find_terms(query: str, terminology: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    matches = []
    for turkish_term in sorted(terminology, key=len, reverse=True):
        if _phrase_matches(query, turkish_term):
            matches.append((turkish_term, terminology[turkish_term]))
    return matches


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def rewrite_query_to_english(query: str, matched_terms: List[Tuple[str, List[str]]]) -> str:
    if ollama is None:
        raise RuntimeError("ollama paketi bulunamadı. English rewrite için Ollama gerekli.")

    terminology_hints = _unique(
        english_term
        for _, equivalents in matched_terms
        for english_term in equivalents
    )
    hint_text = ", ".join(terminology_hints[:30]) if terminology_hints else "none"

    prompt = f"""Rewrite the following Turkish physiotherapy/rehabilitation question into precise clinical English for information retrieval.

Rules:
- Preserve the exact clinical intent.
- Preserve condition, population, intervention, outcome, progression, dosage, safety, and comparison details.
- Preserve explicit numbers, units, time periods, and abbreviations exactly.
- Use standard clinical physiotherapy terminology.
- Prefer terminology used in English clinical guidelines.
- Do not answer the question.
- Do not add information that is not present in the Turkish question.
- Return ONLY the rewritten English question, with no explanation or quotation marks.

Turkish question:
{query}

Useful terminology hints from the FTR dictionary:
{hint_text}
"""

    response = ollama.chat(
        model=REWRITE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical terminology-aware query rewriting assistant. Rewrite only; never answer or add facts. Return only the rewritten English query."
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0, "num_predict": 120},
        think=False,
    )

    rewritten = response["message"]["content"].strip()
    rewritten = rewritten.strip('"')
    if not rewritten:
        raise RuntimeError("English query rewrite boş döndü.")

    rewritten = _preserve_terminology_anchors(rewritten, matched_terms)
    return rewritten


def _contains_term(text: str, term: str) -> bool:
    q = normalize(text)
    t = normalize(term)
    if not t:
        return False
    escaped = re.escape(t)
    return re.search(
        rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",
        q,
        flags=re.UNICODE,
    ) is not None


def _preserve_terminology_anchors(
    rewritten: str,
    matched_terms: List[Tuple[str, List[str]]],
) -> str:
    anchors = []
    for _, equivalents in matched_terms:
        equivalents = _unique(equivalents)
        if not equivalents:
            continue
        if not any(_contains_term(rewritten, term) for term in equivalents):
            anchors.append(equivalents[0])

    if not anchors:
        return rewritten.strip()

    return f"{rewritten.strip()} {' '.join(_unique(anchors))}".strip()

def expand_query(
    query: str,
    terminology: Dict[str, List[str]] | None = None,
    matched_terms: List[Tuple[str, List[str]]] | None = None,
) -> str:
    if terminology is None:
        terminology = load_terminology()
    if matched_terms is None:
        matched_terms = find_terms(query, terminology)

    english_terms = [
        english_term
        for _, equivalents in matched_terms
        for english_term in equivalents
    ]
    unique_terms = _unique(english_terms)
    if not unique_terms:
        return query.strip()
    return f"{query.strip()} {' '.join(unique_terms)}"


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
    added_terms = _unique([
        english_term
        for _, equivalents in matches
        for english_term in equivalents
    ])

    if use_english_rewrite:
        rewritten_query = rewrite_query_to_english(original_query, matches)
    else:
        rewritten_query = original_query

    bm25_query = " ".join(_unique([
        rewritten_query,
        *added_terms,
    ]))

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "expanded_query": bm25_query,
        "matched_terms": [term for term, _ in matches],
        "added_terms": added_terms,
    }