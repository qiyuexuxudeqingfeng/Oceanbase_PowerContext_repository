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

"""Serializable context packs and prefix-reuse deltas."""

# ruff: noqa: TRY003

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


class ContextPackError(ValueError):
    pass


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field} must be a Base64 string array")
    items: list[bytes] = []
    try:
        for encoded in value:
            if not isinstance(encoded, str):
                raise ContextPackError(f"{field} must be a Base64 string array")
            decoded = base64.b64decode(encoded, validate=True)
            if base64.b64encode(decoded).decode("ascii") != encoded:
                raise ContextPackError(f"{field} must use canonical Base64")
            items.append(decoded)
    except (binascii.Error, ValueError) as error:
        raise ContextPackError(f"invalid {field}") from error
    return tuple(items)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": _PACK_SCHEMA, "items": _encode_items(items)}
    return "sha256:" + hashlib.sha256(_json_bytes(unsigned)).hexdigest()


def _pack_wire(items: tuple[bytes, ...], digest: str, byte_count: int) -> bytes:
    return _json_bytes({
        "schema": _PACK_SCHEMA,
        "items": _encode_items(items),
        "digest": digest,
        "byte_count": byte_count,
    })


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    byte_count = 0
    while True:
        updated = len(_pack_wire(items, digest, byte_count))
        if updated == byte_count:
            return byte_count
        byte_count = updated


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(type(item) is not bytes for item in self.items):
            raise ContextPackError("pack items must be a bytes tuple")
        if self.digest != _pack_digest(self.items):
            raise ContextPackError("pack digest does not match its items")
        if type(self.byte_count) is not int or self.byte_count != len(self.to_bytes()):
            raise ContextPackError("pack byte_count does not match its wire size")

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        if not isinstance(items, tuple) or any(type(item) is not bytes for item in items):
            raise ContextPackError("pack items must be a bytes tuple")
        digest = _pack_digest(items)
        return cls(items=items, digest=digest, byte_count=_pack_byte_count(items, digest))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be text")
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        items: list[bytes] = []
        for line in content.splitlines(keepends=True):
            current: list[str] = []
            size = 0
            for character in line:
                encoded_size = len(character.encode("utf-8"))
                if encoded_size > chunk_bytes:
                    raise ContextPackError("chunk_bytes cannot fit a complete UTF-8 character")
                if size + encoded_size > chunk_bytes:
                    items.append("".join(current).encode("utf-8"))
                    current = []
                    size = 0
                current.append(character)
                size += encoded_size
            if current:
                items.append("".join(current).encode("utf-8"))
        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        return _pack_wire(self.items, self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, content: bytes) -> ContextPack:
        payload = _decode_json(content)
        if set(payload) != {"schema", "items", "digest", "byte_count"} or payload["schema"] != _PACK_SCHEMA:
            raise ContextPackError("invalid context pack fields")
        pack = cls.build(_decode_items(payload["items"], "items"))
        if payload["digest"] != pack.digest or payload["byte_count"] != pack.byte_count:
            raise ContextPackError("context pack validation failed")
        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_digest, str) or not _DIGEST.fullmatch(self.base_digest):
            raise ContextPackError("invalid base digest")
        if not isinstance(self.target_digest, str) or not _DIGEST.fullmatch(self.target_digest):
            raise ContextPackError("invalid target digest")
        if type(self.prefix_items) is not int or self.prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")
        if not isinstance(self.appended, tuple) or any(type(item) is not bytes for item in self.appended):
            raise ContextPackError("appended must be a bytes tuple")

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta requires two valid context packs")
        prefix = 0
        for base_item, target_item in zip(base.items, target.items, strict=False):
            if base_item != target_item:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _json_bytes({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encode_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, content: bytes) -> ContextPackDelta:
        payload = _decode_json(content)
        if set(payload) != {"schema", "base_digest", "target_digest", "prefix_items", "appended"}:
            raise ContextPackError("invalid context delta fields")
        if payload["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("invalid context delta schema")
        return cls(
            base_digest=payload["base_digest"],
            target_digest=payload["target_digest"],
            prefix_items=payload["prefix_items"],
            appended=_decode_items(payload["appended"], "appended"),
        )


def _decode_json(content: bytes) -> dict[str, Any]:
    if not isinstance(content, bytes):
        raise ContextPackError("wire content must be bytes")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContextPackError("invalid context wire JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("context wire JSON must be an object")
    return payload


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a context pack and delta")
    if base.digest != delta.base_digest:
        raise ContextPackError("delta base digest does not match the supplied pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return target


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
