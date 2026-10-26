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

"""Durable evidence availability graphs for generation and audit."""

# ruff: noqa: TRY003

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    insert,
    select,
    update,
)

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    pass


class SourceLifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    revision: int = Field(gt=0, strict=True)
    sequence: int = Field(gt=0, strict=True)
    available: bool = Field(strict=True)
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text must not be blank")
        return value


@dataclass(frozen=True, slots=True)
class _LifecycleNode:
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    parents: tuple[str, ...]
    policy: Literal["all", "any"] | None
    revision: int
    sequence: int
    available: bool
    reason: str


_TABLE = Table(
    "pc_source_lifecycle_nodes",
    MetaData(),
    Column("scope_id", String(64), nullable=False),
    Column("node_id", String(256), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("reference", JSON, nullable=False),
    Column("parents", JSON, nullable=False),
    Column("policy", String(3), nullable=True),
    Column("revision", BigInteger, nullable=False),
    Column("sequence", BigInteger, nullable=False),
    Column("available", Boolean, nullable=False),
    Column("reason", String(1024), nullable=False),
    PrimaryKeyConstraint("scope_id", "node_id"),
)


def _reference(kind: str, payload: dict) -> SourceRef | MemoryCitation | ArtifactRef:
    if kind == "source":
        return SourceRef.model_validate(payload)
    if kind == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


def _node_from_row(row) -> _LifecycleNode:
    kind = str(row["kind"])
    return _LifecycleNode(
        node_id=str(row["node_id"]),
        kind=kind,
        reference=_reference(kind, dict(row["reference"])),
        parents=tuple(row["parents"]),
        policy=row["policy"],
        revision=int(row["revision"]),
        sequence=int(row["sequence"]),
        available=bool(row["available"]),
        reason=str(row["reason"]),
    )


def _availability(node: _LifecycleNode, nodes: dict[str, _LifecycleNode], memo: dict[str, bool]) -> bool:
    if node.node_id in memo:
        return memo[node.node_id]
    if node.kind == "source":
        return node.available
    support = [_availability(nodes[parent], nodes, memo) for parent in node.parents]
    memo[node.node_id] = all(support) if node.policy == "all" else any(support)
    return memo[node.node_id]


class SourceLifecycleService:
    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._database = runtime._provider.database
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with self._database.transaction() as connection:
            await connection.run_sync(lambda sync: _TABLE.create(sync, checkfirst=True))
        self._ready = True

    async def _nodes(self) -> dict[str, _LifecycleNode]:
        await self._ensure()
        async with self._database.transaction() as connection:
            rows = (await connection.execute(select(_TABLE).where(_TABLE.c.scope_id == self.scope_id))).mappings()
        return {str(row["node_id"]): _node_from_row(row) for row in rows}

    async def _insert(self, node: _LifecycleNode) -> None:
        async with self._database.transaction() as connection:
            await connection.execute(
                insert(_TABLE).values(
                    scope_id=self.scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    reference=node.reference.model_dump(mode="json"),
                    parents=list(node.parents),
                    policy=node.policy,
                    revision=node.revision,
                    sequence=node.sequence,
                    available=node.available,
                    reason=node.reason,
                )
            )

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int) -> None:
        self._validate_node_id(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("source revision must be a positive integer")
        await self._runtime.records.for_scope(self.scope_id).get_source(source_ref.source_type, source_ref.source_id)
        nodes = await self._nodes()
        existing = nodes.get(node_id)
        if existing is not None:
            if existing.kind != "source" or existing.reference != source_ref or existing.revision != revision:
                raise SourceLifecycleError("source registration conflicts with existing state")
            return
        await self._insert(_LifecycleNode(node_id, "source", source_ref, (), None, revision, 0, True, ""))

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        self._validate_derivative(node_id, parents, policy)
        async with self._runtime._context(self.scope_id) as context:
            await context.artifacts.memory.validate_citation(citation)
        nodes = await self._nodes()
        self._validate_cycle(node_id, parents, nodes)
        existing = nodes.get(node_id)
        if existing is not None:
            if (
                existing.kind != "memory"
                or existing.reference != citation
                or existing.parents != parents
                or existing.policy != policy
            ):
                raise SourceLifecycleError("memory registration conflicts with existing state")
        else:
            available = _availability(
                _LifecycleNode(node_id, "memory", citation, parents, policy, 1, 0, False, ""), nodes, {}
            )
            node = _LifecycleNode(node_id, "memory", citation, parents, policy, 1, 0, available, "")
            await self._insert(node)
            nodes[node_id] = node
        await self._sync_memory(nodes[node_id], nodes)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        self._validate_derivative(node_id, parents, policy)
        if artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("topic registration requires a Topic Memory reference")
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision
        )
        nodes = await self._nodes()
        self._validate_cycle(node_id, parents, nodes)
        existing = nodes.get(node_id)
        if existing is not None:
            if (
                existing.kind != "topic"
                or existing.reference != artifact_ref
                or existing.parents != parents
                or existing.policy != policy
            ):
                raise SourceLifecycleError("topic registration conflicts with existing state")
            return
        available = _availability(
            _LifecycleNode(node_id, "topic", artifact_ref, parents, policy, 1, 0, False, ""), nodes, {}
        )
        await self._insert(_LifecycleNode(node_id, "topic", artifact_ref, parents, policy, 1, 0, available, ""))

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        nodes = await self._nodes()
        source = nodes.get(event.node_id)
        if source is None or source.kind != "source" or source.revision != event.revision:
            raise SourceLifecycleError("unknown Source or lifecycle revision mismatch")
        if event.sequence < source.sequence:
            raise SourceLifecycleError("lifecycle sequence is not monotonic")
        if event.sequence == source.sequence:
            if source.available == event.available and source.reason == event.reason:
                return self._active_ids(nodes)
            raise SourceLifecycleError("lifecycle event conflicts with existing event")

        changed = {
            node_id: _LifecycleNode(
                node.node_id,
                node.kind,
                node.reference,
                node.parents,
                node.policy,
                node.revision,
                event.sequence if node_id == event.node_id else node.sequence,
                event.available if node_id == event.node_id else node.available,
                event.reason if node_id == event.node_id else node.reason,
            )
            for node_id, node in nodes.items()
        }
        memo: dict[str, bool] = {}
        updated = {
            node_id: _LifecycleNode(
                node.node_id,
                node.kind,
                node.reference,
                node.parents,
                node.policy,
                node.revision,
                node.sequence,
                _availability(node, changed, memo),
                node.reason,
            )
            for node_id, node in changed.items()
        }
        for node in updated.values():
            if node.kind == "memory":
                await self._sync_memory(node, updated)
        async with self._database.transaction() as connection:
            for node in updated.values():
                await connection.execute(
                    update(_TABLE)
                    .where(_TABLE.c.scope_id == self.scope_id, _TABLE.c.node_id == node.node_id)
                    .values(available=node.available, sequence=node.sequence, reason=node.reason)
                )
        return self._active_ids(updated)

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        nodes = await self._nodes()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "audit":
            return node.reference
        if purpose == "generation":
            if not node.available:
                raise SourceLifecycleError("lifecycle node is unavailable for generation")
            return node.reference
        raise SourceLifecycleError("unknown lifecycle purpose")

    async def _sync_memory(self, node: _LifecycleNode, nodes: dict[str, _LifecycleNode]) -> None:
        available = _availability(node, nodes, {})
        citation = node.reference
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory lifecycle reference is invalid")
        async with self._runtime._context(self.scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            manifest = next(entry for entry in current.content.manifest.entries if entry.entry_id == citation.entry_id)
            target_state = "active" if available else "inactive"
            if manifest.state == target_state:
                return
            entry = next(entry for entry in await service.entries(current) if entry.entry_id == citation.entry_id)
            if available:
                await service.reactivate(current, entries=(entry,), reason=node.reason or None)
            else:
                await service.forget(current, entries=(entry,), reason=node.reason or None)

    async def _filter_topic_result(self, result):
        nodes = await self._nodes()
        blocked = {
            node.reference.artifact_id
            for node in nodes.values()
            if node.kind == "topic" and not node.available and isinstance(node.reference, ArtifactRef)
        }
        if not blocked:
            return result
        return result.model_copy(
            update={"hits": tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in blocked)}
        )

    @staticmethod
    def _active_ids(nodes: dict[str, _LifecycleNode]) -> list[str]:
        return sorted(node.node_id for node in nodes.values() if node.available)

    @staticmethod
    def _validate_node_id(node_id: str) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must not be blank")

    @staticmethod
    def _validate_derivative(node_id: str, parents: tuple[str, ...], policy: str) -> None:
        SourceLifecycleService._validate_node_id(node_id)
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be all or any")
        if not parents or len(set(parents)) != len(parents) or any(not parent.strip() for parent in parents):
            raise SourceLifecycleError("derived nodes require unique registered parents")

    @staticmethod
    def _validate_cycle(node_id: str, parents: tuple[str, ...], nodes: dict[str, _LifecycleNode]) -> None:
        if node_id in parents or any(parent not in nodes for parent in parents):
            raise SourceLifecycleError("missing parent or dependency cycle")
        graph = {node: value.parents for node, value in nodes.items()}
        graph[node_id] = parents
        visiting: set[str] = set()

        def visit(current: str) -> None:
            if current in visiting:
                raise SourceLifecycleError("dependency cycle")
            visiting.add(current)
            for parent in graph[current]:
                visit(parent)
            visiting.remove(current)

        visit(node_id)


class SourceLifecycleApplication:
    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime
        self._services: dict[str, SourceLifecycleService] = {}

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        if scope_id not in self._services:
            self._services[scope_id] = SourceLifecycleService(self._runtime, scope_id)
        return self._services[scope_id]


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent", "SourceLifecycleService"]
