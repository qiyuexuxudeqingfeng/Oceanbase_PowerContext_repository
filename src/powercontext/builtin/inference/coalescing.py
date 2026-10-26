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

"""Cross-call embedding batching, cancellation isolation, and LRU reuse."""

# ruff: noqa: TRY003

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


class _EmbeddingBatch:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.completion: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.vectors: dict[str, EmbeddingVector] = {}
        self.usage = InferenceUsage(requests=0)
        self.error: Exception | None = None
        self.started = False


class CoalescingEmbeddingModel:
    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if type(coalesce_ms) not in {int, float} or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be non-negative")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._tasks: set[asyncio.Task[object]] = set()
        self._inflight: dict[str, _EmbeddingBatch] = {}
        self._current: _EmbeddingBatch | None = None
        self._lock = asyncio.Lock()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0, input_tokens=0, output_tokens=0))
        unique_texts = tuple(dict.fromkeys(texts))
        cached: dict[str, EmbeddingVector] = {}
        assignments: dict[str, _EmbeddingBatch] = {}
        async with self._lock:
            for text in unique_texts:
                if text in self._cache:
                    self._cache.move_to_end(text)
                    cached[text] = self._cache[text]
                    continue
                batch = self._inflight.get(text)
                if batch is None:
                    batch = self._current
                    if batch is None or len(batch.texts) >= self._max_batch_size:
                        batch = _EmbeddingBatch(())
                        self._current = batch
                        task = asyncio.create_task(self._close_after_window(batch))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
                    batch.texts = (*batch.texts, text)
                    self._inflight[text] = batch
                    if len(batch.texts) >= self._max_batch_size:
                        self._start_locked(batch)
                assignments[text] = batch

        batches = tuple(dict.fromkeys(assignments.values()))
        for batch in batches:
            await asyncio.shield(batch.completion)
            if batch.error is not None:
                raise batch.error

        vectors = tuple(cached[text] if text in cached else assignments[text].vectors[text] for text in texts)
        return EmbeddingResult(vectors=vectors, usage=_combined_usage(batch.usage for batch in batches))

    async def _close_after_window(self, batch: _EmbeddingBatch) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        async with self._lock:
            if self._current is batch and not batch.started:
                self._start_locked(batch)

    def _start_locked(self, batch: _EmbeddingBatch) -> None:
        if self._current is batch:
            self._current = None
        if batch.started:
            return
        batch.started = True
        task = asyncio.create_task(self._run(batch))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, batch: _EmbeddingBatch) -> None:
        await asyncio.sleep(0)
        try:
            result = await self._model.embed(batch.texts)
            _validate_embedding_result(batch, result)
            batch.vectors = dict(zip(batch.texts, result.vectors, strict=True))
            batch.usage = result.usage
        except Exception as error:
            batch.error = error

        async with self._lock:
            for text in batch.texts:
                if self._inflight.get(text) is batch:
                    del self._inflight[text]
                if batch.error is None and self._cache_size:
                    self._cache[text] = batch.vectors[text]
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
            if not batch.completion.done():
                batch.completion.set_result(None)


def _validate_embedding_result(batch: _EmbeddingBatch, result: EmbeddingResult) -> None:
    if len(result.vectors) != len(batch.texts):
        raise ValueError("embedding provider returned a different vector count")


def _combined_usage(usages) -> InferenceUsage:
    values = tuple(usages)
    input_values = [usage.input_tokens for usage in values if usage.input_tokens is not None]
    output_values = [usage.output_tokens for usage in values if usage.output_tokens is not None]
    return InferenceUsage(
        requests=sum(usage.requests for usage in values),
        input_tokens=sum(input_values) if input_values else (0 if values else None),
        output_tokens=sum(output_values) if output_values else (0 if values else None),
    )


__all__ = ["CoalescingEmbeddingModel"]
