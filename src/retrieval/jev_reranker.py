
from __future__ import annotations

import os
import time
from typing import Any

import requests


JEV_API_URL = "https://jev-ai.pro/api/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_MAX_QUESTIONS = 64


class JevReranker:
    """Jev API ile tek istekte birden fazla adayı puanlayan reranker."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        timeout: float = 90.0,
    ):
        self.api_key = api_key or os.getenv("JEV_AI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "JEV_AI_API_KEY ortam değişkeni bulunamadı."
            )

        self.model = model
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self.request_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_usage: dict[str, Any] = {}

    @staticmethod
    def _extract_score(answer: Any, question_id: str) -> float:
        """Jev cevabından noul skorunu güvenli biçimde çıkarır."""

        if isinstance(answer, dict):
            score = answer.get("noul")
        else:
            score = None

        if score is None:
            raise ValueError(
                f"{question_id} için noul skoru bulunamadı. "
                f"Cevap: {answer!r}"
            )

        score = float(score)

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{question_id} için beklenmeyen skor: {score}"
            )

        return score

    def _score_candidates(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[float]:
        """Bütün adayları tek API isteğinde puanlar."""

        if not candidates:
            return []

        if len(candidates) > JEV_MAX_QUESTIONS:
            raise ValueError(
                f"Tek istekte en fazla {JEV_MAX_QUESTIONS} aday "
                f"puanlanabilir; verilen: {len(candidates)}"
            )

        # Her aday için state içinde ayrı bir alan ve
        # questions içinde aynı isimli bir puanlama sorusu oluştur.
        state: dict[str, str] = {"query": query}
        questions: dict[str, dict[str, str]] = {}
        candidate_ids: list[str] = []

        for index, item in enumerate(candidates, start=1):
            question_id = f"candidate_{index:02d}"
            candidate_ids.append(question_id)

            state[question_id] = str(item.get("document", ""))

            questions[question_id] = {
                "type": "noul",
                "instructions": (
                    f"Evaluate the relevance of {question_id} to the query. "
                    "Does this candidate chunk directly support answering "
                    "the query? Return a score from 0 to 1, where 0 means "
                    "not relevant and 1 means directly and strongly "
                    "supports answering the query. Use only the query and "
                    "the specified candidate chunk."
                ),
            }

        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

        started = time.perf_counter()

        response = requests.post(
            JEV_API_URL,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.request_count += 1

        print(
            f"  Jev API: HTTP {response.status_code} | "
            f"{len(candidates)} aday | {elapsed_ms:.0f} ms"
        )

        if not response.ok:
            # API anahtarını veya Authorization başlığını yazdırma.
            print("  Jev hata yanıtı:", response.text[:2000])
            response.raise_for_status()

        data = response.json()
        answers = data.get("answers")

        if not isinstance(answers, dict):
            raise ValueError(
                f"Jev yanıtında 'answers' nesnesi bulunamadı: {data}"
            )

        usage = data.get("usage") or {}
        self.last_usage = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "elapsed_ms": round(elapsed_ms, 2),
            "candidate_count": len(candidates),
        }

        if isinstance(usage.get("input_tokens"), (int, float)):
            self.total_input_tokens += int(usage["input_tokens"])

        if isinstance(usage.get("output_tokens"), (int, float)):
            self.total_output_tokens += int(usage["output_tokens"])

        print(
            "  Jev usage:",
            f"input={self.last_usage['input_tokens']},",
            f"output={self.last_usage['output_tokens']}",
        )

        scores = []

        # Eksik bir skor varsa sessizce devam etme:
        # aday sırası ile skor eşleşmesi bozulmamalı.
        for question_id in candidate_ids:
            if question_id not in answers:
                raise ValueError(
                    f"Jev yanıtında {question_id} eksik. "
                    f"Gelen anahtarlar: {list(answers.keys())}"
                )

            scores.append(
                self._extract_score(
                    answers[question_id],
                    question_id,
                )
            )

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Aynı arayüz: adayları puanlar ve en iyi top_k sonucu döndürür."""

        if not candidates:
            self.last_usage = {}
            return []

        top_k = max(1, min(top_k, len(candidates)))

        # Her rerank çağrısında yalnızca bir API isteği.
        scores = self._score_candidates(query, candidates)

        ranked = []

        for original_rank, (item, score) in enumerate(
            zip(candidates, scores),
            start=1,
        ):
            result = dict(item)
            result["jev_score"] = score
            result["pre_rerank_rank"] = original_rank
            ranked.append(result)

        # Eşit skor durumunda önceki RRF sırasını koru.
        ranked.sort(
            key=lambda item: (
                -item["jev_score"],
                item["pre_rerank_rank"],
            )
        )

        for rank, item in enumerate(ranked, start=1):
            item["jev_rank"] = rank

        return ranked[:top_k]
