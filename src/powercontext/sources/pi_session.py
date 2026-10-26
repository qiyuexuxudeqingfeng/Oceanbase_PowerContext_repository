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

"""Immutable Pi session evidence and incremental history capture."""

# ruff: noqa: TRY003, TRY004

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    Source,
    SourceMaterialization,
    TextEvidence,
)

PI_EVENT_SOURCE_NAME = "pi-event"


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def _identity(session_id: str, event_id: str) -> str:
    if not session_id or not event_id:
        raise ValueError("Pi session and event identities must be non-empty")
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def _digest(event: PiEventInput) -> str:
    encoded = json.dumps(
        event.record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_parts(content: JsonValue, *, include_tool_calls: bool = False) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        raise ValueError("Pi message content must be a string or a list")

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Pi message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text blocks must contain string text")
            parts.append(text)
        elif block_type == "toolCall" and include_tool_calls:
            name = block.get("name")
            call_id = block.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError("Pi tool calls must contain string names and IDs")
            arguments = json.dumps(
                block.get("arguments"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            parts.append(f"[tool_call {name} {call_id}] {arguments}")
    return parts


def _event_from_record(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:  # noqa: C901
    event_type = record.get("type")
    if not isinstance(event_type, str):
        return None
    event_id = record["id"]
    parent_id = record["parentId"]
    if not isinstance(event_id, str):
        raise ValueError("Pi event IDs must be strings")

    if event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError("Pi summaries must be strings")
        body = summary
    elif event_type == "custom_message":
        role = "custom"
        body = "\n".join(_text_parts(record.get("content")))
    elif event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message events must contain a message object")
        message_role = message.get("role")
        if not isinstance(message_role, str) or not message_role:
            raise ValueError("Pi messages must contain a non-empty role")
        role = message_role
        if message_role == "toolResult":
            text = "\n".join(_text_parts(message.get("content")))
            if not text:
                return None
            tool_name = message.get("toolName")
            call_id = message.get("toolCallId")
            if not isinstance(tool_name, str) or not isinstance(call_id, str):
                raise ValueError("Pi tool results must contain tool names and call IDs")
            body = f"[tool_result {tool_name} {call_id}] {text}"
        elif message_role == "bashExecution":
            command = message.get("command")
            output = message.get("output")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi bash events must contain string commands and output")
            if not command and not output:
                return None
            body = f"$ {command}\n{output}"
        else:
            body = "\n".join(_text_parts(message.get("content"), include_tool_calls=True))
    else:
        return None

    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _parse_archive(content: bytes) -> tuple[str, dict[str, dict[str, JsonValue]], dict[str, PiEventInput]]:  # noqa: C901
    try:
        text = content.decode("utf-8-sig")
        records = [json.loads(line, parse_constant=_reject_json_constant) for line in text.splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Pi session archive") from error
    if not records:
        raise ValueError("Pi session archive is empty")

    header = records[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session header must be an object")
    session_id = header.get("id")
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session header must have type session and version 3")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session ID must be a non-empty string")

    nodes: dict[str, dict[str, JsonValue]] = {}
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be objects")
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in nodes:
            raise ValueError("Pi event IDs must be non-empty and unique")
        if "parentId" not in record:
            raise ValueError("Pi events must contain parentId")
        parent_id = record["parentId"]
        if parent_id is not None and (not isinstance(parent_id, str) or parent_id not in nodes):
            raise ValueError("Pi event parents must refer to earlier nodes")
        nodes[event_id] = record

    events = {
        event_id: event
        for event_id, record in nodes.items()
        if (event := _event_from_record(session_id, record)) is not None
    }
    return session_id, nodes, events


def _selected_events(
    nodes: dict[str, dict[str, JsonValue]],
    events: dict[str, PiEventInput],
    leaf_id: str | None,
) -> tuple[PiEventInput, ...]:
    if not nodes:
        raise ValueError("Pi session contains no events")
    selected_leaf = leaf_id if leaf_id is not None else next(reversed(nodes))
    if not isinstance(selected_leaf, str) or selected_leaf not in nodes:
        raise ValueError("Pi session leaf does not exist")

    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = selected_leaf
    while current is not None:
        if current in seen or current not in nodes:
            raise ValueError("Pi session contains a cycle or missing parent")
        seen.add(current)
        chain.append(current)
        parent = nodes[current]["parentId"]
        current = parent if isinstance(parent, str) else None
    chain.reverse()
    return tuple(events[event_id] for event_id in chain if event_id in events)


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    _, nodes, events = _parse_archive(value.content)
    return _selected_events(nodes, events, value.leaf_id)


class PiEventSourceAdapter:
    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = _identity(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventSourceAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        _, nodes, events = _parse_archive(self.path.read_bytes())
        selected = _selected_events(nodes, events, self.leaf_id)
        previous = self._checkpoint_events(session.checkpoint)
        available = {_identity(event.session_id, event.event_id): _digest(event) for event in events.values()}
        for identity, digest in previous.items():
            if identity not in available:
                raise ValueError("a previously imported Pi event is missing")
            if available[identity] != digest:
                raise ValueError("a previously imported Pi event was modified")

        updated = dict(previous)
        for event in selected:
            identity = _identity(event.session_id, event.event_id)
            digest = _digest(event)
            if identity in updated:
                if updated[identity] != digest:
                    raise ValueError("a previously imported Pi event was modified")
                continue
            await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            updated[identity] = digest
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"version": 1, "events": updated},
        )

    @staticmethod
    def _checkpoint_events(checkpoint: JsonValue) -> dict[str, str]:
        if checkpoint is None:
            return {}
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
            raise ValueError("invalid Pi session checkpoint")
        events = checkpoint.get("events")
        if not isinstance(events, dict) or not all(
            isinstance(identity, str) and isinstance(digest, str) for identity, digest in events.items()
        ):
            raise ValueError("invalid Pi session checkpoint events")
        return dict(events)


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
]
