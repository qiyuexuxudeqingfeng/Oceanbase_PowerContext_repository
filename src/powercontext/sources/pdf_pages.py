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

"""Physical PDF page evidence and scoped Memory retrieval."""

# ruff: noqa: TRY003

from __future__ import annotations

import hashlib
import io
import re
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.search import analyze_fts_query
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceMaterialization,
    TextEvidence,
)

PDF_PAGE_SOURCE_NAME = "pdf-page"
_QUESTION_STOP_WORDS = frozenset([
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "calculate",
    "defines",
    "define",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "please",
    "respond",
    "show",
    "state",
    "the",
    "then",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
])

_PROBE_GROUP_SIZE = 2
_PROBE_TERM_CAP = 24
_PROBE_CANDIDATE_LIMIT = 80


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def _identity(document_id: str, document_sha256: str, page_number: int) -> str:
    identity = f"pdf:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"PDF page source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    if not value.document_id.strip():
        raise ValueError("document_id must not be blank")
    document_sha256 = hashlib.sha256(value.content).hexdigest()
    _identity(value.document_id, document_sha256, 1)
    try:
        reader = PdfReader(io.BytesIO(value.content))
    except Exception as error:
        raise ValueError("invalid or unreadable PDF document") from error
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError("PDF requires a password")
    try:
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=page.extract_text() or "",
            )
            for page_number, page in enumerate(reader.pages, start=1)
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("invalid or unreadable PDF document") from error
    for page in pages:
        _identity(page.document_id, page.document_sha256, page.page_number)
    return pages


class PDFPageSourceAdapter:
    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        if not value.document_id.strip():
            raise ValueError("document_id must not be blank")
        if type(value.page_number) is not int or value.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if re.fullmatch(r"[0-9a-f]{64}", value.document_sha256) is None:
            raise ValueError("document_sha256 must be a lowercase SHA-256 digest")
        identity = _identity(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=identity,
            source_id=identity,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
            materialization=SourceMaterialization.CAPTURED,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidenceProjection(),),
)


def _hit_key(hit: MemoryHit) -> tuple[str, str, int, str, str]:
    ref = hit.memory_ref
    return ref.family, ref.artifact_id, ref.revision, hit.entry_id, hit.entry_version_id


def _citation(hit: MemoryHit) -> MemoryCitation:
    return MemoryCitation(
        memory_ref=hit.memory_ref,
        entry_id=hit.entry_id,
        entry_version_id=hit.entry_version_id,
    )


def _probe_queries(question: str, size: int = _PROBE_GROUP_SIZE, cap: int = _PROBE_TERM_CAP) -> tuple[str, ...]:
    """Split the question into short lexical probes.

    A long question is rejected by the Memory API's coverage admission, and the page
    that carries the answer often repeats only a couple of its tokens, so the whole
    sentence is never used as a query. Short probes keep the admission floor low and
    let the reciprocal-rank fusion below rank pages by how many probes they satisfy.
    """

    terms = [term for term in analyze_fts_query(question).split() if term not in _QUESTION_STOP_WORDS]
    queries = [
        " ".join(terms[start : start + size])
        for start in range(0, min(len(terms), cap), size)
        if terms[start : start + size]
    ]
    if not queries:
        return (question.strip(),) if question.strip() else ()
    return tuple(dict.fromkeys(queries))


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not question.strip():
        raise ValueError("question must not be blank")
    queries = _probe_queries(question)
    candidates: dict[tuple[str, str, int, str, str], tuple[MemoryHit, float]] = {}
    for query in queries:
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_PROBE_CANDIDATE_LIMIT))
        for rank, hit in enumerate(page.hits, start=1):
            key = _hit_key(hit)
            score = 1.0 / (60 + rank)
            if key in candidates:
                previous, previous_score = candidates[key]
                candidates[key] = (previous, previous_score + score)
            else:
                candidates[key] = (hit, score)

    ordered = sorted(candidates.values(), key=lambda item: -item[1])
    citations: list[MemoryCitation] = []
    seen_sources: set[tuple[str, str]] = set()
    for hit, _ in ordered:
        citation = _citation(hit)
        entry = await memory.get(GetMemoryEntryRequest(citation=citation))
        sources = {(source.source_type, source.source_id) for source in entry.entry.sources}
        if not sources or sources & seen_sources:
            continue
        seen_sources.update(sources)
        citations.append(citation)
        if len(citations) == limit:
            break
    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
