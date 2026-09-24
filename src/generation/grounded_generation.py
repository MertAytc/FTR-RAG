from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json
import re

try:
    import ollama
except ImportError:
    ollama = None

GENERATION_MODEL = "qwen2.5:7b-instruct"

ENGLISH_SYSTEM_PROMPT = """You are a physiotherapy and rehabilitation evidence assistant.
Answer ONLY from the supplied CONTEXT.

STRICT RULES:
1. Use only claims explicitly supported by CONTEXT.
2. Do not use outside medical knowledge or guesses.
3. Never invent sets, repetitions, duration, frequency, intensity, progression criteria, or other clinical values.
4. If a requested fact is absent, say it is not specified in the context.
5. Preserve clinical terminology and numerical values from the source.
6. Put [SOURCE X] immediately after every important clinical claim.
7. Do not attach a source to unsupported claims.
8. Do not recommend medication or medication doses.
9. Do not diagnose.
10. Do not strengthen evidence beyond what the source states.
11. If sources conflict, report the conflict instead of resolving it yourself.
12. Do not fill missing parts of the question with unrelated context.
13. Answer the user's actual question type. If the user asks for an exercise/treatment plan, you MAY construct a physiotherapy exercise plan from the supplied context, but every exercise, parameter, progression rule, and safety condition must be supported by the context.
14. If a requested plan parameter is not supported by the context, explicitly mark that parameter as not specified rather than inventing a value.
15. Medication recommendations and medication doses are always prohibited. Exercise dosage is allowed when it is explicitly supported by the context.
16. Do not diagnose or prescribe medication.
17. Answer in precise clinical English.
18. Use at most 4 short bullets.
19. Do not interpret corrupted or unclear extracted text as clinical fact.
"""
SYSTEM_PROMPT = ENGLISH_SYSTEM_PROMPT
TRANSLATION_SYSTEM_PROMPT = """Translate the supplied grounded physiotherapy/rehabilitation answer from English to Turkish.

This is a CLINICAL FTR translation task, not a general-purpose translation task.
Use standard Turkish physiotherapy and rehabilitation terminology and preserve the
clinical meaning exactly. Do not translate technical FTR terms into unrelated
everyday words.

STRICT TRANSLATION RULES:
1. Translate only; do not add, remove, infer, strengthen, weaken, or change clinical information.
2. Preserve every number, unit, dosage, frequency, duration, abbreviation, exercise name, and named clinical term.
3. Preserve every [SOURCE X] citation exactly and in the same location.
4. Do not create new citations.
5. Do not introduce new medical advice.
6. Keep the same bullet structure and meaning.
7. Use natural, professional Turkish used by physiotherapists and rehabilitation clinicians.
8. Prefer established FTR terminology from the supplied terminology glossary when available.
9. Examples of terminology behavior: "exercise progression" -> "egzersiz progresyonu" or "egzersizin ilerletilmesi"; "progression criteria" -> "progresyon kriterleri"; "exercise" -> "egzersiz"; "motor learning" -> "motor öğrenme"; "repetitive training" -> "tekrarlı eğitim" or "tekrarlı motor eğitim"; "range of motion (ROM)" -> "eklem hareket açıklığı (ROM)".
10. Never translate "exercise progression" as "hamle gelişimi" or another unrelated everyday expression.
11. If an English clinical term has no reliable Turkish equivalent in the glossary, keep the English clinical term in parentheses rather than inventing a misleading translation.
12. Medication recommendations and medication doses are never to be added.
13. Return ONLY the Turkish translation.
"""

TERMINOLOGY_PATH = Path(__file__).resolve().parents[1] / "retrieval" / "clinical_terminology.json"

def _load_translation_glossary(query: str) -> str:
    """Load matching Turkish->English FTR terminology from the project's JSON glossary."""
    try:
        from src.retrieval.query_processor import find_terms, load_terminology
        terminology = load_terminology(TERMINOLOGY_PATH)
        matches = find_terms(query, terminology)
        lines = []
        for turkish_term, equivalents in matches:
            lines.append(f"- {turkish_term} => {', '.join(equivalents)}")
        return "\n".join(lines) if lines else "No matching glossary terms found."
    except (FileNotFoundError, json.JSONDecodeError, OSError, ImportError, TypeError):
        return "Glossary unavailable; use standard Turkish physiotherapy terminology and do not invent translations."

OUT_OF_SCOPE_PATTERNS = [r"\bilaç\b", r"\bilac\b", r"\bilaç\s+dozu\b", r"\bilac\s+dozu\b", r"\btanı\s+koy\b"]
INSUFFICIENT_EVIDENCE_TEXT = "Mevcut kaynaklarda bu soruyu güvenilir biçimde yanıtlamak için yeterli bilgi bulunmuyor. Kaynaklarda yer almayan bilgileri tahmin ederek tamamlamıyorum."


@dataclass
class GenerationResult:
    answer: str
    grounded: bool
    insufficient_evidence: bool
    out_of_scope: bool
    sources: List[Dict]
    prompt: str
    english_answer: str = ""
    english_query: str = ""


def is_out_of_scope(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return any(re.search(pattern, normalized) for pattern in OUT_OF_SCOPE_PATTERNS)


def build_english_prompt(english_query: str, context_text: str) -> str:
    return (
        f"{ENGLISH_SYSTEM_PROMPT}\n\n"
        f"CLINICAL QUESTION:\n{english_query}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        "Answer the clinical question using only the context."
    )


def _source_numbers_from_answer(answer: str) -> List[int]:
    return sorted({
        int(number)
        for group in re.findall(r"\[SOURCE\s+([0-9]+(?:\s*,\s*[0-9]+)*)\]", answer, flags=re.I)
        for number in re.findall(r"\d+", group)
    })


def validate_citations(answer: str, source_count: int) -> bool:
    cited = _source_numbers_from_answer(answer)
    return bool(cited) and all(1 <= number <= source_count for number in cited)


def _citation_structure(answer: str) -> List[str]:
    return re.findall(r"\[SOURCE\s+[0-9]+(?:\s*,\s*[0-9]+)*\]", answer, flags=re.I)


def _translate_to_turkish(
    english_answer: str,
    model: str,
    original_query: str,
) -> str:
    glossary = _load_translation_glossary(original_query)
    user_prompt = (
        f"FTR TERMINOLOGY GLOSSARY (matched to the original Turkish question):\n"
        f"{glossary}\n\n"
        "TRANSLATE THE FOLLOWING GROUNDED ENGLISH ANSWER TO TURKISH. "
        "Use the glossary where applicable. Preserve all [SOURCE X] tags exactly.\n\n"
        f"{english_answer}"
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0, "num_predict": 350},
    )
    return response["message"]["content"].strip()


def generate_grounded_answer(
    query: str,
    context_result: Dict,
    model: str = GENERATION_MODEL,
    english_query: str | None = None,
) -> GenerationResult:
    sources = context_result.get("sources", [])
    context_text = context_result.get("context_text", "")

    if is_out_of_scope(query):
        answer = "Bu soru mevcut FTR-RAG kapsamının dışında veya güvenli biçimde yanıtlanması için uygun değil. İlaç, ilaç dozu veya kaynaklarda yer almayan klinik kararları tahmin ederek önermiyorum."
        return GenerationResult(answer, False, False, True, sources, "", english_query or "")

    if not sources or not context_text.strip():
        return GenerationResult(INSUFFICIENT_EVIDENCE_TEXT, False, True, False, [], "", english_query or "")

    if ollama is None:
        raise RuntimeError("ollama paketi bulunamadı.")

    if english_query is None:
        # Fallback only. In the normal pipeline the rewrite from query_processor
        # should be passed in so the question is rewritten exactly once.
        from src.retrieval.query_processor import process_query
        english_query = process_query(query)["rewritten_query"]

    prompt = build_english_prompt(english_query, context_text)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ENGLISH_SYSTEM_PROMPT},
            {"role": "user", "content": f"CLINICAL QUESTION:\n{english_query}\n\nCONTEXT:\n{context_text}\n\nAnswer only from the context."},
        ],
        options={"temperature": 0.0, "num_predict": 300},
    )

    english_answer = response["message"]["content"].strip()
    grounded = validate_citations(english_answer, len(sources))

    # If the grounded English answer has no valid source syntax, do not let the
    # translation stage hide that failure.
    if not grounded:
        answer = english_answer + "\n\nNot: Yanıtta geçerli bir [SOURCE X] kaynağı bulunmadığı için kaynak bağlılığı doğrulanamadı."
        return GenerationResult(answer, False, False, False, sources, prompt, english_answer, english_query)

    turkish_answer = _translate_to_turkish(english_answer, model, query)

    # Translation must preserve citation structure exactly. If it changes or
    # removes citations, return the English answer with an explicit warning rather
    # than silently presenting an apparently grounded Turkish answer.
    if _citation_structure(turkish_answer) != _citation_structure(english_answer):
        answer = english_answer + "\n\nNot: Türkçe çeviri kaynak etiketlerini koruyamadığı için doğrulanmış çeviri sunulmadı."
        return GenerationResult(answer, False, False, False, sources, prompt, english_answer, english_query)

    return GenerationResult(turkish_answer, True, False, False, sources, prompt, english_answer, english_query)
