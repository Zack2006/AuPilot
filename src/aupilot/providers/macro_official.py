from __future__ import annotations

from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from aupilot.core.hashing import canonical_json_sha256
from aupilot.macro_gate.schemas import OFFICIAL_DOMAINS, MacroDocument

PARSER_VERSION = "official-html-rss-v1"


def _validate_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_DOMAINS:
        raise ValueError("Refusing a non-official macro source")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._title_complete = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        if tag == "title" and not self._title_complete:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            if self._in_title:
                self._title_complete = True
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        self.parts.append(cleaned)
        if self._in_title:
            self.title_parts.append(cleaned)


def extract_official_text(payload: str) -> tuple[str, str | None]:
    """Extract searchable text and the first title from HTML or RSS/XML."""
    extractor = _TextExtractor()
    extractor.feed(payload)
    content = "\n".join(extractor.parts)
    title = " ".join(extractor.title_parts) or None
    return content, title


def fetch_official_document(
    *,
    source: str,
    event_type: str,
    url: str,
    timeout_seconds: float = 20.0,
) -> MacroDocument:
    _validate_official_url(url)
    retrieved = datetime.now(UTC)
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "AuPilot/0.1 read-only macro evidence collector"},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
    final_url = str(response.url)
    _validate_official_url(final_url)
    content, extracted_title = extract_official_text(response.text)
    title = extracted_title or f"{source} {event_type} official page"
    if len(content) < 100:
        raise RuntimeError("Official macro page contained insufficient text")
    content_sha = canonical_json_sha256({"url": final_url, "content": content})
    return MacroDocument(
        doc_id=f"{source}:{event_type}:{content_sha[:20]}",
        source=source,
        provider_id=source,
        source_tier="A",
        event_type=event_type,
        title=title,
        canonical_url=final_url,
        content=content,
        published_at_utc=None,
        retrieved_at_utc=retrieved,
        eligible_from_utc=retrieved,
        content_sha256=content_sha,
        replay_eligible=False,
    )
