# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batch HTTP scoring with global, identity-preserving Memory reranking."""

# ruff: noqa: TRY003

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision, _validate_bounds
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if type(limit) is not int:
            raise ValueError("rerank limit must be an integer")
        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        request_count = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(enumerate(candidates, start=1))[start : start + self._batch_size]
            expected_ranks = {rank for rank, _ in batch}
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
                },
            )
            request_count += 1
            response.raise_for_status()
            batch_scores = self._parse_response(response, expected_ranks)
            scores.update(batch_scores)

        selected = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=request_count))

    @staticmethod
    def _parse_response(response: httpx.Response, expected_ranks: set[int]) -> dict[int, float]:
        try:
            payload = response.json()
        except (UnicodeError, ValueError) as error:
            raise ValueError("reranker response must be valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
            raise ValueError("reranker response must contain only scores")

        scores: dict[int, float] = {}
        for item in payload["scores"]:
            if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                raise ValueError("reranker score records must contain only rank and score")
            rank = item["rank"]
            score = item["score"]
            if type(rank) is not int or rank not in expected_ranks or rank in scores:
                raise ValueError("reranker response must cover each batch rank exactly once")
            if type(score) not in {int, float} or not math.isfinite(score):
                raise ValueError("reranker scores must be finite numbers")
            scores[rank] = float(score)
        if set(scores) != expected_ranks:
            raise ValueError("reranker response must cover each batch rank exactly once")
        return scores


__all__ = ["HTTPMemoryReranker"]
