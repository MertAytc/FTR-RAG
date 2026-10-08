from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


JEV_API_URL = "https://jev-ai.pro/api/v1/systemone"
JEV_MODEL = "jev-latest"


class JevReranker:
    """Jev-based reranker for an already-retrieved candidate pool."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        timeout: float = 30.0,
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

    def _score_candidate(
        self,
        query: str,
        document: str,
    ) -> float:

        payload = {
            "model": self.model,
            "state": {
                "query": query,
                "candidate_chunk": document,
            },
            "questions": {
                "relevant": {
                    "type": "noul",
                    "instructions": (
                        "Does the candidate chunk directly support "
                        "answering the query?"
                    ),
                }
            },
        }

        response = requests.post(
            JEV_API_URL,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )

        print("STATUS:", response.status_code)

        if not response.ok:
            print("RESPONSE:", response.text)

        response.raise_for_status()

        data = response.json()

        answer = data["answers"]["relevant"]

        if isinstance(answer, dict) and "noul" in answer:
            return float(answer["noul"])

        raise ValueError(
            f"Jev response içinde 'noul' bulunamadı: {data}"
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        top_k = max(
            1,
            min(top_k, len(candidates)),
        )

        ranked = []

        for original_rank, item in enumerate(
            candidates,
            start=1,
        ):
            document = str(
                item.get("document", "")
            )

            score = self._score_candidate(
                query=query,
                document=document,
            )

            result = dict(item)

            result["jev_score"] = score
            result["pre_rerank_rank"] = original_rank

            ranked.append(result)

        ranked.sort(
            key=lambda item: item["jev_score"],
            reverse=True,
        )

        for rank, item in enumerate(
            ranked,
            start=1,
        ):
            item["jev_rank"] = rank

        return ranked[:top_k]