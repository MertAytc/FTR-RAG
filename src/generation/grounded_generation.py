from dataclasses import dataclass
from typing import Dict, List
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
INSUFFICIENT_EVIDENCE_TEXT = "The available sources do not contain enough information to answer this question reliably. I will not guess information that is not present in the sources."


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
    patterns = (r"\bmedication\b", r"\bdrug\s+dose\b", r"\bmedication\s+dose\b", r"\bdiagnos(?:e|is|ing)\b", r"\bprescrib(?:e|ing|ed)\b")
    return any(re.search(pattern, normalized) for pattern in patterns)


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



def generate_grounded_answer(
    query: str,
    context_result: Dict,
    model: str = GENERATION_MODEL,
    english_query: str | None = None,
) -> GenerationResult:
    sources = context_result.get("sources", [])
    context_text = context_result.get("context_text", "")

    if is_out_of_scope(query):
        answer = "This request is outside the supported FTR-RAG scope. I cannot recommend medication, medication doses, diagnoses, or unsupported clinical decisions."
        return GenerationResult(answer, False, False, True, sources, "", english_query or "")

    if not sources or not context_text.strip():
        return GenerationResult(INSUFFICIENT_EVIDENCE_TEXT, False, True, False, [], "", english_query or "")

    if ollama is None:
        raise RuntimeError("ollama paketi bulunamadı.")

    if english_query is None:
        # Questions are supplied in English by the benchmark/user.
        english_query = query.strip()

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
        answer = english_answer + "\n\nNote: Source grounding could not be verified because the answer contains no valid [SOURCE X] citation."
        return GenerationResult(answer, False, False, False, sources, prompt, english_answer, english_query)

    return GenerationResult(english_answer, True, False, False, sources, prompt, english_answer, english_query)
