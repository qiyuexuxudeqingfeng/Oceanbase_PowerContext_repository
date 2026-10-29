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
import math
import re
from collections import Counter
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

_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_TERM_SUFFIXES = ("ings", "ing", "ies", "ied", "es", "ed", "s")
_BM25_K1 = 1.2
_BM25_B = 0.15
_STATEMENT_WEIGHT = 1.6
_CONCEPT_WEIGHT = 0.6
_FUSION_WEIGHT = 1e-6

# A financial question usually names the statement that carries the answer, but the
# page itself often writes only the statement title, so the title is matched on the
# question's vocabulary instead of its wording.
_STATEMENT_PATTERNS = {
    "income": re.compile(r"statement[s]?\s+of\s+(income|operations|earnings)", re.IGNORECASE),
    "balance": re.compile(r"balance\s+sheet|statement[s]?\s+of\s+financial\s+(position|condition)", re.IGNORECASE),
    "cashflow": re.compile(r"statement[s]?\s+of\s+cash\s*flow", re.IGNORECASE),
    "equity": re.compile(
        r"statement[s]?\s+of\s+(stockholders|shareholders|changes\s+in)\b.{0,24}equity", re.IGNORECASE
    ),
}

_STATEMENT_PROBES = {
    "income": "statement income",
    "balance": "balance sheet",
    "cashflow": "cash flows",
    "equity": "shareholders equity",
}

_STATEMENT_CUES = {
    "income": "margin revenue revenues sales income earnings profit operating gross expense cost ebitda tax",
    "balance": (
        "asset liabilities liability debt balance liquidity current ratio working capital inventory "
        "receivable payable equity goodwill intangible shares treasury capital intensive leverage"
    ),
    "cashflow": (
        "cash flow capex capital expenditure depreciation amortization buyback repurchase dividend "
        "investment investing financing"
    ),
    "equity": "stockholders shareholders retained comprehensive treasury",
}

# Report pages are indexed by whatever phrase the filing uses, and a question asks about
# the concept instead: "legal battles" never appears next to "legal proceedings".
_CONCEPTS = {
    "legal": (
        "legal litigation lawsuit battle proceeding",
        ("legal proceeding", "litigation", "legal matter", "legal contingenc", "lawsuits"),
    ),
    "segment": (
        "segment categories category",
        ("business segment", "segment data", "reportable segment", "segment information"),
    ),
    "geograph": (
        "geograph region country countries international",
        ("geographic", "region", "international"),
    ),
    "customer": ("customer client", ("customer",)),
    "cyclical": ("cyclical seasonal cycle", ("cyclical", "seasonal")),
    "employee": ("employee headcount workforce", ("employee", "headcount")),
    "competition": ("competitor competition competitive", ("competition", "competitor")),
    "liquidity": (
        "liquidity working capital current ratio",
        ("liquidity", "working capital", "current assets"),
    ),
    "debt": ("debt borrowing leverage", ("long-term debt", "borrowings", "debt")),
    "pension": ("pension retiree postretirement", ("pension", "postretirement")),
    "tax": ("tax", ("effective tax rate", "income taxes", "provision for income tax")),
    "inventory": ("inventory inventorie turnover", ("inventory", "inventories")),
    "goodwill": ("goodwill intangible impairment", ("goodwill", "intangible")),
    "buyback": ("repurchase buyback", ("repurchase", "treasury")),
    "restructuring": ("restructur", ("restructuring", "severance")),
    "research": ("research development", ("research and development",)),
}


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


def _fold(term: str) -> str:
    """Fold a term to a comparison stem so plurals and inflections meet."""

    for suffix in _TERM_SUFFIXES:
        if len(term) > 4 and term.endswith(suffix):
            stem = term[: -len(suffix)]
            return f"{stem}y" if suffix in {"ies", "ied"} else stem
    return term


def _terms(text: str) -> list[str]:
    return [_fold(term) for term in _TOKEN_PATTERN.findall(text.lower())]


def _probe_queries(question: str, size: int = _PROBE_GROUP_SIZE, cap: int = _PROBE_TERM_CAP) -> tuple[str, ...]:
    """Split the question into short lexical probes.

    A long question is rejected by the Memory API's coverage admission, and the page
    that carries the answer often repeats only a couple of its tokens, so the whole
    sentence is never used as a query. The probes only widen the candidate pool; the
    ranking below decides which pages survive.
    """

    terms = [term for term in analyze_fts_query(question).split() if term not in _QUESTION_STOP_WORDS]
    queries = [
        " ".join(terms[start : start + size])
        for start in range(0, min(len(terms), cap), size)
        if terms[start : start + size]
    ]
    queries.extend(_support_queries(question))
    if not queries:
        return (question.strip(),) if question.strip() else ()
    return tuple(dict.fromkeys(queries))


def _implied_statements(question: str) -> dict[str, int]:
    """Count how strongly the question points at each primary statement."""

    lowered = question.lower()
    weights = {}
    for name, cues in _STATEMENT_CUES.items():
        hits = sum(1 for cue in cues.split() if cue in lowered)
        if hits:
            weights[name] = hits
    for name, phrase in (
        ("cashflow", "cash flow"),
        ("balance", "balance sheet"),
        ("income", "income statement"),
        ("income", "statement of operations"),
    ):
        if phrase in lowered:
            weights[name] = weights.get(name, 0) + 3
    return weights


def _implied_concepts(question: str) -> tuple[tuple[str, ...], ...]:
    lowered = question.lower()
    return [phrases for cues, phrases in _CONCEPTS.values() if any(cue in lowered for cue in cues.split())]


def _support_queries(question: str) -> list[str]:
    """Extra probes for the pages a lexical match usually misses.

    Statement titles and report phrases are written differently from the question,
    so they are queried directly instead of being left to chance. Probes stay at two
    terms because the Memory API rejects longer queries for low coverage.
    """

    phrases = [_STATEMENT_PROBES[name] for name in _implied_statements(question)]
    phrases.extend(group[0] for group in _implied_concepts(question))
    return [" ".join(phrase.split()[:2]) for phrase in phrases]


def _bm25_scores(question: str, texts: list[str]) -> list[float]:
    terms = [term for term in _terms(question) if term not in _QUESTION_STOP_WORDS]
    if not terms:
        return [0.0] * len(texts)
    page_terms = [Counter(_terms(text)) for text in texts]
    lengths = [sum(counter.values()) for counter in page_terms]
    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    doc_freq = Counter()
    for counter in page_terms:
        doc_freq.update(counter.keys())
    total = len(texts)
    scores = [0.0] * total
    for term in dict.fromkeys(terms):
        frequency = doc_freq.get(term, 0)
        if not frequency:
            continue
        idf = math.log((total - frequency + 0.5) / (frequency + 0.5) + 1.0)
        for index, counter in enumerate(page_terms):
            tf = counter.get(term, 0)
            if not tf:
                continue
            norm = 1 - _BM25_B + _BM25_B * (lengths[index] / (average or 1.0))
            scores[index] += idf * tf * (_BM25_K1 + 1) / (tf + _BM25_K1 * norm)
    return scores


def _statement_boost(texts: list[str], weights: dict[str, int]) -> list[float]:
    boost = [0.0] * len(texts)
    if not weights:
        return boost
    lowered = [text[:800] for text in texts]
    for name, weight in weights.items():
        pattern = _STATEMENT_PATTERNS[name]
        add = _STATEMENT_WEIGHT * min(weight, 3) / 3
        for index, head in enumerate(lowered):
            if pattern.search(head):
                boost[index] += add
    return boost


def _concept_boost(texts: list[str], phrases: tuple[tuple[str, ...], ...]) -> list[float]:
    boost = [0.0] * len(texts)
    if not phrases:
        return boost
    lowered = [text.lower() for text in texts]
    for group in phrases:
        for index, text in enumerate(lowered):
            count = sum(text.count(phrase) for phrase in group)
            if count:
                boost[index] += _CONCEPT_WEIGHT * min(count, 3) / 3
    return boost


def _rank_hits(
    question: str,
    candidates: dict[tuple[str, str, int, str, str], tuple[MemoryHit, float]],
) -> list[MemoryHit]:
    """Rank the fused candidate pool with the evidence-page signals.

    Reciprocal-rank fusion rewards a page for satisfying a probe, which favours pages
    that repeat the question's commonest wording. A BM25 pass over the pool instead
    weighs rare terms, and the statement and report-phrase boosts cover the pages whose
    wording never matches the question at all.
    """

    hits = [hit for hit, _ in candidates.values()]
    fusion = [score for _, score in candidates.values()]
    texts = [hit.text for hit in hits]
    lexical = _bm25_scores(question, texts)
    peak = max(lexical) or 1.0
    statement = _statement_boost(texts, _implied_statements(question))
    concept = _concept_boost(texts, _implied_concepts(question))
    peak_fusion = max(fusion) or 1.0
    scored = [
        lexical[index] / peak + statement[index] + concept[index] + _FUSION_WEIGHT * fusion[index] / peak_fusion
        for index in range(len(hits))
    ]
    return [hit for _, hit in sorted(zip(scored, hits, strict=True), key=lambda item: -item[0])]


async def _collect_candidates(memory, question: str) -> dict[tuple[str, str, int, str, str], tuple[MemoryHit, float]]:
    """Gather the pool the ranking works on, fused across every probe.

    No single probe returns the page that carries the answer, so the pool is the union
    of all of them and the reciprocal-rank sum below only breaks ties later.
    """

    candidates: dict[tuple[str, str, int, str, str], tuple[MemoryHit, float]] = {}
    for query in _probe_queries(question):
        page = await memory.search(SearchMemoryRequest(query=query, mode="fts", limit=_PROBE_CANDIDATE_LIMIT))
        for rank, hit in enumerate(page.hits, start=1):
            key = _hit_key(hit)
            score = 1.0 / (60 + rank)
            if key in candidates:
                previous, previous_score = candidates[key]
                candidates[key] = (previous, previous_score + score)
            else:
                candidates[key] = (hit, score)
    return candidates


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not question.strip():
        raise ValueError("question must not be blank")
    candidates = await _collect_candidates(memory, question)
    if not candidates:
        return ()

    citations: list[MemoryCitation] = []
    seen_sources: set[tuple[str, str]] = set()
    for hit in _rank_hits(question, candidates):
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
