from time import perf_counter
from typing import Dict, Generator, Any
import inspect
import ollama

from src.generation.grounded_generation import GENERATION_MODEL, SYSTEM_PROMPT


def stream_grounded_generation(
    query: str,
    context_text: str,
    language: str = "Turkish",
    temperature: float = 0.0,
    num_predict: int = 300,
    model: str | None = None,
) -> Generator[Dict[str, Any], None, None]:
    """Stream a grounded answer and expose TTFT/generation timing."""
    model = model or GENERATION_MODEL

    if language == "English":
        instruction = (
            "Answer in English. Use only the CONTEXT. "
            "Use at most 4 short bullet points. "
            "Put [SOURCE X] after each important clinical claim. "
            "If the requested information is not explicitly in the CONTEXT, "
            "say that it is not specified and do not guess."
        )
    else:
        instruction = (
            "Yanıtı Türkçe ver. Yalnızca CONTEXT'i kullan. "
            "En fazla 4 kısa madde kullan. "
            "Her önemli klinik iddianın sonunda [SOURCE X] kullan. "
            "İstenen bilgi CONTEXT'te açıkça yoksa belirtilmediğini söyle ve tahmin etme."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"KULLANICI SORUSU:\n{query}\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"{instruction}"
            ),
        },
    ]

    started = perf_counter()
    first_token_time = None
    token_count = 0
    answer_parts = []

    try:
        chat_kwargs = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "stream": True,
        }

        # Newer Ollama clients expose `think`. Disable visible reasoning for
        # thinking-capable models such as Qwen3 so the UI waits for the actual
        # answer instead of appearing empty while the model reasons.
        try:
            if "think" in inspect.signature(ollama.chat).parameters:
                chat_kwargs["think"] = False
        except (TypeError, ValueError):
            pass

        try:
            response = ollama.chat(**chat_kwargs)
        except TypeError as exc:
            # Backward compatibility with older Ollama Python clients.
            if "think" in chat_kwargs:
                chat_kwargs.pop("think", None)
                response = ollama.chat(**chat_kwargs)
            else:
                raise exc

        for part in response:
            token = (part.get("message") or {}).get("content", "")
            if token:
                now = perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    token_count = 1
                    yield {
                        "event": "first_token",
                        "ttft_ms": round((now - started) * 1000, 2),
                        "token": token,
                        "token_count": token_count,
                    }
                else:
                    token_count += 1
                    yield {"event": "token", "token": token, "token_count": token_count}
                answer_parts.append(token)

            if part.get("done"):
                ended = perf_counter()
                ttft_ms = ((first_token_time - started) * 1000) if first_token_time else None
                generation_ms = ((ended - first_token_time) * 1000) if first_token_time else ((ended - started) * 1000)
                answer = "".join(answer_parts).strip()

                if not answer:
                    yield {
                        "event": "error",
                        "model": model,
                        "error": "Model çalıştı ancak görünür bir cevap üretmedi. Seçilen modelin Ollama'da kurulu ve chat için kullanılabilir olduğunu kontrol edin.",
                        "total_ms": round((ended - started) * 1000, 2),
                        "token_count": token_count,
                    }
                    return

                yield {
                    "event": "done",
                    "model": model,
                    "answer": answer,
                    "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round((ended - started) * 1000, 2),
                    "token_count": token_count,
                }
    except Exception as exc:
        yield {
            "event": "error",
            "model": model,
            "error": str(exc),
            "total_ms": round((perf_counter() - started) * 1000, 2),
            "token_count": token_count,
        }
