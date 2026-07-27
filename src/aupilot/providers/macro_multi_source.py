from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import httpx

from aupilot.core.hashing import canonical_json_sha256
from aupilot.macro_gate.schemas import (
    ALLOWED_EVIDENCE_DOMAINS,
    MacroClaim,
    MacroDocument,
    MacroObservation,
)

from .macro_official import extract_official_text


USER_AGENT = "AuPilot/0.2 official-macro-evidence-collector"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class ProviderFetchError(RuntimeError):
    def __init__(self, reason_code: str, *, retryable: bool) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.retryable = retryable


@dataclass(frozen=True)
class RetrievalRequest:
    as_of_utc: datetime

    def __post_init__(self) -> None:
        if self.as_of_utc.tzinfo is None or self.as_of_utc.utcoffset() is None:
            raise ValueError("RetrievalRequest.as_of_utc must be timezone-aware")


@dataclass(frozen=True)
class ProviderBatch:
    provider_id: str
    source_tier: str
    required: bool
    started_at_utc: datetime
    completed_at_utc: datetime
    fetch_succeeded: bool
    documents: tuple[MacroDocument, ...] = ()
    observations: tuple[MacroObservation, ...] = ()
    claims: tuple[MacroClaim, ...] = ()
    raw_response_sha256: tuple[str, ...] = ()
    error_type: str | None = None
    reason_code: str | None = None
    retryable: bool = False

    def manifest_record(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "source_tier": self.source_tier,
            "required": self.required,
            "started_at_utc": self.started_at_utc.isoformat(),
            "completed_at_utc": self.completed_at_utc.isoformat(),
            "fetch_succeeded": self.fetch_succeeded,
            "document_count": len(self.documents),
            "observation_count": len(self.observations),
            "claim_count": len(self.claims),
            "raw_response_sha256": list(self.raw_response_sha256),
            "error_type": self.error_type,
            "reason_code": self.reason_code,
            "retryable": self.retryable,
        }


class MacroProvider(Protocol):
    provider_id: str
    source_tier: str
    required: bool

    def fetch(self, request: RetrievalRequest) -> ProviderBatch: ...


@dataclass(frozen=True)
class DisabledProvider:
    """Fail-closed provider placeholder that performs no network request."""

    provider_id: str
    source_tier: str = "A"
    required: bool = True

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        del request
        raise ProviderFetchError("SOURCE_DISABLED_BY_LOCAL_SETTINGS", retryable=False)


@dataclass(frozen=True)
class HttpPayload:
    url: str
    content: bytes
    content_type: str
    sha256: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class SafeHttpClient:
    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
        connect_timeout_seconds: float = 5.0,
        total_timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        self.transport = transport
        self.sleep = sleep
        self.timeout = httpx.Timeout(total_timeout_seconds, connect=connect_timeout_seconds)
        self.max_attempts = max_attempts
        self.max_response_bytes = max_response_bytes

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EVIDENCE_DOMAINS:
            raise ProviderFetchError("UNTRUSTED_PROVIDER_URL", retryable=False)

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json_payload: dict[str, Any] | None = None,
        allowed_mime_prefixes: tuple[str, ...] = (
            "application/json",
            "application/xml",
            "application/atom+xml",
            "application/rss+xml",
            "text/xml",
            "text/html",
            "text/plain",
        ),
    ) -> HttpPayload:
        self._validate_url(url)
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    transport=self.transport,
                    headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
                ) as client:
                    response = client.request(
                        method,
                        url,
                        params=params,
                        json=json_payload,
                    )
                self._validate_url(str(response.url))
                if response.status_code in {401, 403}:
                    raise ProviderFetchError(
                        f"HTTP_{response.status_code}", retryable=False
                    )
                if response.status_code == 429:
                    error = ProviderFetchError("HTTP_429", retryable=True)
                    if attempt + 1 == self.max_attempts:
                        raise error
                    retry_after = response.headers.get("Retry-After", "")
                    delay = min(float(retry_after), 30.0) if retry_after.isdigit() else 2**attempt
                    self.sleep(delay)
                    continue
                if response.status_code >= 500:
                    error = ProviderFetchError(
                        f"HTTP_{response.status_code}", retryable=True
                    )
                    if attempt + 1 == self.max_attempts:
                        raise error
                    self.sleep(2**attempt)
                    continue
                response.raise_for_status()
                content = response.content
                if len(content) > self.max_response_bytes:
                    raise ProviderFetchError("RESPONSE_TOO_LARGE", retryable=False)
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not any(content_type.startswith(value) for value in allowed_mime_prefixes):
                    raise ProviderFetchError("UNEXPECTED_MIME_TYPE", retryable=False)
                return HttpPayload(
                    url=str(response.url),
                    content=content,
                    content_type=content_type,
                    sha256=hashlib.sha256(content).hexdigest().upper(),
                )
            except ProviderFetchError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = error
                if attempt + 1 < self.max_attempts:
                    self.sleep(2**attempt)
                    continue
        raise ProviderFetchError(
            type(last_error).__name__ if last_error else "NETWORK_FAILURE",
            retryable=True,
        )

    def get(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        allowed_mime_prefixes: tuple[str, ...] = (
            "application/json",
            "application/xml",
            "application/atom+xml",
            "application/rss+xml",
            "text/xml",
            "text/html",
            "text/plain",
        ),
    ) -> HttpPayload:
        return self._request(
            "GET",
            url,
            params=params,
            allowed_mime_prefixes=allowed_mime_prefixes,
        )

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        allowed_mime_prefixes: tuple[str, ...] = ("application/json",),
    ) -> HttpPayload:
        return self._request(
            "POST",
            url,
            json_payload=payload,
            allowed_mime_prefixes=allowed_mime_prefixes,
        )


def _observed_at(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _observation_and_claim(
    *,
    provider_id: str,
    metric_id: str,
    slot: str,
    event_type: str,
    observation_date: date,
    value: float,
    unit: str,
    source_url: str,
    payload_sha256: str,
    retrieved_at_utc: datetime,
    display_text: str,
    revision_status: str = "INITIAL",
) -> tuple[MacroObservation, MacroClaim]:
    identity = canonical_json_sha256(
        {
            "provider_id": provider_id,
            "metric_id": metric_id,
            "date": observation_date.isoformat(),
            "value": value,
            "payload": payload_sha256,
        }
    )
    observation_id = (
        f"{provider_id}:{metric_id}:{observation_date.isoformat()}:{identity[:20]}"
    )
    observation = MacroObservation(
        observation_id=observation_id,
        series_id=metric_id,
        metric_id=metric_id,
        provider_id=provider_id,
        source_tier="A",
        observation_date=observation_date,
        value=value,
        realtime_start=observation_date,
        realtime_end=observation_date,
        eligible_from_utc=retrieved_at_utc,
        retrieved_at_utc=retrieved_at_utc,
        first_seen_at_utc=retrieved_at_utc,
        source_url=source_url,
        source_payload_sha256=payload_sha256,
        initial_release_only=revision_status == "INITIAL",
        retrieval_method="API",
        official_primary=True,
        revision_status=revision_status,
        unit=unit,
        independence_key=provider_id,
    )
    claim = MacroClaim(
        claim_id=f"claim:{identity[:32]}",
        slot=slot,
        event_type=event_type,
        claim_type=slot.split(".", 1)[1],
        normalized_value=f"{observation_date.isoformat()}:{value:.8g}:{unit}",
        display_text=display_text,
        value=value,
        unit=unit,
        reference_period=observation_date.isoformat(),
        observed_at_utc=_observed_at(observation_date),
        source_record_id=observation_id,
        provider_id=provider_id,
        source_tier="A",
        canonical_url=source_url,
        first_seen_at_utc=retrieved_at_utc,
        retrieved_at_utc=retrieved_at_utc,
        eligible_from_utc=retrieved_at_utc,
        content_sha256=payload_sha256,
        retrieval_method="API",
        official_primary=True,
        revision_status=revision_status,
        independence_key=provider_id,
    )
    return observation, claim


class NewYorkFedRatesProvider:
    provider_id = "new_york_fed_effr"
    source_tier = "A"
    required = True

    def __init__(self, endpoint: str, *, http: SafeHttpClient | None = None) -> None:
        self.endpoint = endpoint
        self.http = http or SafeHttpClient()

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        started = datetime.now(UTC)
        payload = self.http.get(self.endpoint, allowed_mime_prefixes=("application/json",))
        data = json.loads(payload.text)
        rates = data.get("refRates")
        if not isinstance(rates, list) or not rates:
            raise ProviderFetchError("NYFED_EFFR_EMPTY", retryable=False)
        row = max(rates, key=lambda item: str(item.get("effectiveDate", "")))
        observed = date.fromisoformat(str(row["effectiveDate"]))
        value = float(row["percentRate"])
        observation, claim = _observation_and_claim(
            provider_id=self.provider_id,
            metric_id="EFFR",
            slot="RATES.effr",
            event_type="RATES",
            observation_date=observed,
            value=value,
            unit="percent",
            source_url=self.endpoint,
            payload_sha256=payload.sha256,
            retrieved_at_utc=started,
            display_text=f"New York Fed EFFR was {value:g}% for {observed.isoformat()}.",
            revision_status=("REVISED" if row.get("revisionIndicator") else "INITIAL"),
        )
        return ProviderBatch(
            provider_id=self.provider_id,
            source_tier=self.source_tier,
            required=self.required,
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            fetch_succeeded=True,
            observations=(observation,),
            claims=(claim,),
            raw_response_sha256=(payload.sha256,),
        )


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _treasury_rows(payload: HttpPayload) -> list[dict[str, str]]:
    root = ET.fromstring(payload.content)
    rows: list[dict[str, str]] = []
    for properties in root.iter():
        if _local_name(properties) != "properties":
            continue
        row = {_local_name(child): (child.text or "").strip() for child in properties}
        if row.get("NEW_DATE"):
            rows.append(row)
    if not rows:
        raise ProviderFetchError("TREASURY_XML_EMPTY", retryable=False)
    return rows


class TreasuryCurveProvider:
    provider_id = "us_treasury_curve"
    source_tier = "A"
    required = True

    def __init__(
        self,
        nominal_endpoint: str,
        real_endpoint: str,
        *,
        http: SafeHttpClient | None = None,
    ) -> None:
        self.nominal_endpoint = nominal_endpoint
        self.real_endpoint = real_endpoint
        self.http = http or SafeHttpClient()

    @staticmethod
    def _latest(rows: list[dict[str, str]], as_of: date) -> tuple[date, dict[str, str]]:
        candidates = [
            (date.fromisoformat(row["NEW_DATE"][:10]), row)
            for row in rows
            if date.fromisoformat(row["NEW_DATE"][:10]) <= as_of
        ]
        if not candidates:
            raise ProviderFetchError("TREASURY_NO_POINT_IN_TIME_ROW", retryable=False)
        return max(candidates, key=lambda item: item[0])

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        started = datetime.now(UTC)
        year = str(request.as_of_utc.year)
        nominal = self.http.get(
            self.nominal_endpoint,
            params={"data": "daily_treasury_yield_curve", "field_tdr_date_value": year},
        )
        real = self.http.get(
            self.real_endpoint,
            params={"data": "daily_treasury_real_yield_curve", "field_tdr_date_value": year},
        )
        nominal_date, nominal_row = self._latest(
            _treasury_rows(nominal), request.as_of_utc.date()
        )
        real_date, real_row = self._latest(_treasury_rows(real), request.as_of_utc.date())
        observations: list[MacroObservation] = []
        claims: list[MacroClaim] = []
        specs = (
            ("TREASURY_NOMINAL_2Y_PAR_YIELD", "RATES.nominal_2y", nominal_date, nominal_row["BC_2YEAR"], nominal.sha256, self.nominal_endpoint),
            ("TREASURY_NOMINAL_10Y_PAR_YIELD", "RATES.nominal_10y", nominal_date, nominal_row["BC_10YEAR"], nominal.sha256, self.nominal_endpoint),
            ("TREASURY_REAL_10Y_PAR_YIELD", "RATES.real_10y", real_date, real_row["TC_10YEAR"], real.sha256, self.real_endpoint),
        )
        for metric, slot, observed, raw_value, digest, url in specs:
            value = float(raw_value)
            observation, claim = _observation_and_claim(
                provider_id=self.provider_id,
                metric_id=metric,
                slot=slot,
                event_type="RATES",
                observation_date=observed,
                value=value,
                unit="percent",
                source_url=url,
                payload_sha256=digest,
                retrieved_at_utc=started,
                display_text=f"U.S. Treasury {metric} was {value:g}% for {observed.isoformat()}.",
            )
            observations.append(observation)
            claims.append(claim)
        if nominal_date == real_date:
            proxy = float(nominal_row["BC_10YEAR"]) - float(real_row["TC_10YEAR"])
            proxy_hash = canonical_json_sha256(
                {
                    "nominal_payload": nominal.sha256,
                    "real_payload": real.sha256,
                    "date": nominal_date.isoformat(),
                    "proxy": proxy,
                }
            )
            observation, claim = _observation_and_claim(
                provider_id=self.provider_id,
                metric_id="TREASURY_CMT_10Y_BREAKEVEN_PROXY",
                slot="RATES.breakeven_proxy_10y",
                event_type="RATES",
                observation_date=nominal_date,
                value=proxy,
                unit="percentage_points",
                source_url=self.nominal_endpoint,
                payload_sha256=proxy_hash,
                retrieved_at_utc=started,
                display_text=(
                    "Treasury nominal-minus-real 10Y proxy was "
                    f"{proxy:g} percentage points for {nominal_date.isoformat()}."
                ),
            )
            observations.append(observation)
            claims.append(claim)
        return ProviderBatch(
            provider_id=self.provider_id,
            source_tier=self.source_tier,
            required=self.required,
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            fetch_succeeded=True,
            observations=tuple(observations),
            claims=tuple(claims),
            raw_response_sha256=(nominal.sha256, real.sha256),
        )


class BlsReleaseProvider:
    provider_id = "bls_public_data_api"
    source_tier = "A"
    required = True
    SERIES = {
        "CPI_U": ("CUUR0000SA0", "CPI.latest_initial_release", "CPI", "index"),
        "TOTAL_NONFARM_PAYROLLS": ("CES0000000001", "NFP.latest_initial_release", "NFP", "thousands"),
        "UNEMPLOYMENT_RATE": ("LNS14000000", None, "NFP", "percent"),
        "AVERAGE_HOURLY_EARNINGS": ("CES0500000003", None, "NFP", "usd_per_hour"),
    }

    def __init__(
        self,
        endpoint: str,
        *,
        registration_key: str | None = None,
        http: SafeHttpClient | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.registration_key = (registration_key or "").strip() or None
        self.http = http or SafeHttpClient()

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        started = datetime.now(UTC)
        observations: list[MacroObservation] = []
        claims: list[MacroClaim] = []
        request_payload: dict[str, Any] = {
            "seriesid": [value[0] for value in self.SERIES.values()],
            "startyear": str(max(request.as_of_utc.year - 1, 1913)),
            "endyear": str(request.as_of_utc.year),
        }
        if self.registration_key is not None:
            request_payload["registrationkey"] = self.registration_key
        payload = self.http.post_json(self.endpoint, request_payload)
        body = json.loads(payload.text)
        if body.get("status") != "REQUEST_SUCCEEDED":
            messages = body.get("message", [])
            message = " ".join(str(item) for item in messages).lower()
            quota_exhausted = "daily threshold" in message or "request limit" in message
            raise ProviderFetchError(
                "BLS_API_QUOTA_EXHAUSTED" if quota_exhausted else "BLS_API_REQUEST_FAILED",
                retryable=not quota_exhausted,
            )
        series_rows = body.get("Results", {}).get("series", [])
        by_series_id = {
            str(item.get("seriesID")): item
            for item in series_rows
            if isinstance(item, dict) and item.get("seriesID")
        }
        for metric, (series_id, slot, event_type, unit) in self.SERIES.items():
            series = by_series_id.get(series_id)
            rows = [] if series is None else series.get("data", [])
            latest = next((row for row in rows if str(row.get("latest", "")).lower() == "true"), None)
            if latest is None:
                raise ProviderFetchError(f"BLS_{metric}_LATEST_MISSING", retryable=False)
            period = str(latest["period"])
            if not re.fullmatch(r"M(0[1-9]|1[0-2])", period):
                raise ProviderFetchError("BLS_UNSUPPORTED_PERIOD", retryable=False)
            observed = date(int(latest["year"]), int(period[1:]), 1)
            value = float(latest["value"])
            revision = "INITIAL"
            observation, claim = _observation_and_claim(
                provider_id=self.provider_id,
                metric_id=metric,
                slot=slot or "NFP.latest_initial_release",
                event_type=event_type,
                observation_date=observed,
                value=value,
                unit=unit,
                source_url=self.endpoint,
                payload_sha256=payload.sha256,
                retrieved_at_utc=started,
                display_text=f"BLS {metric} was {value:g} for {latest['periodName']} {latest['year']}.",
                revision_status=revision,
            )
            observations.append(observation)
            if slot is not None:
                claims.append(claim)
        return ProviderBatch(
            provider_id=self.provider_id,
            source_tier=self.source_tier,
            required=self.required,
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            fetch_succeeded=True,
            observations=tuple(observations),
            claims=tuple(claims),
            raw_response_sha256=(payload.sha256,),
        )


class BeaReleaseProvider:
    provider_id = "bea_personal_income_outlays"
    source_tier = "A"
    required = True

    def __init__(self, rss_endpoint: str, *, http: SafeHttpClient | None = None) -> None:
        self.rss_endpoint = rss_endpoint
        self.http = http or SafeHttpClient()

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        started = datetime.now(UTC)
        feed = self.http.get(self.rss_endpoint)
        root = ET.fromstring(feed.content)
        candidates: list[tuple[datetime, str, str]] = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published_raw = (item.findtext("pubDate") or "").strip()
            if "personal income and outlays" not in title.lower() or not link or not published_raw:
                continue
            published = parsedate_to_datetime(published_raw).astimezone(UTC)
            if published <= request.as_of_utc:
                candidates.append((published, title, link))
        if not candidates:
            raise ProviderFetchError("BEA_PCE_RELEASE_MISSING", retryable=False)
        published, title, link = max(candidates, key=lambda item: item[0])
        page = self.http.get(link, allowed_mime_prefixes=("text/html",))
        content, page_title = extract_official_text(page.text)
        if len(content) < 100:
            raise ProviderFetchError("BEA_PCE_RELEASE_TOO_SHORT", retryable=False)
        doc_id = f"bea:PCE:{page.sha256[:20]}"
        document = MacroDocument(
            doc_id=doc_id,
            source=self.provider_id,
            provider_id=self.provider_id,
            source_tier="A",
            event_type="PCE",
            title=page_title or title,
            canonical_url=link,
            content=content,
            published_at_utc=published,
            retrieved_at_utc=started,
            first_seen_at_utc=started,
            eligible_from_utc=started,
            content_sha256=page.sha256,
            retrieval_method="HTML",
            official_primary=True,
            revision_status="INITIAL",
            independence_key=self.provider_id,
        )
        reference_period = title.split(",", 1)[1].strip() if "," in title else title
        claim = MacroClaim(
            claim_id=f"claim:{page.sha256[:32]}",
            slot="PCE.latest_initial_release",
            event_type="PCE",
            claim_type="latest_initial_release",
            normalized_value=f"{published.date().isoformat()}:{reference_period.lower()}",
            display_text=f"BEA published {title}.",
            value={"release_title": title},
            reference_period=reference_period,
            observed_at_utc=published,
            source_record_id=doc_id,
            provider_id=self.provider_id,
            source_tier="A",
            canonical_url=link,
            published_at_utc=published,
            first_seen_at_utc=started,
            retrieved_at_utc=started,
            eligible_from_utc=started,
            content_sha256=page.sha256,
            retrieval_method="HTML",
            official_primary=True,
            revision_status="INITIAL",
            independence_key=self.provider_id,
        )
        return ProviderBatch(
            provider_id=self.provider_id,
            source_tier=self.source_tier,
            required=self.required,
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            fetch_succeeded=True,
            documents=(document,),
            claims=(claim,),
            raw_response_sha256=(feed.sha256, page.sha256),
        )


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


class FedReleaseProvider:
    provider_id = "federal_reserve_fomc_release"
    source_tier = "A"
    required = True
    STATEMENT = re.compile(
        r"(?:fomcstatement|monetary)(\d{8})(?:a)?\.htm$", re.IGNORECASE
    )
    TARGET = re.compile(
        r"target range for the federal funds rate at\s+([0-9./]+)\s+to\s+([0-9./]+)\s+percent",
        re.IGNORECASE,
    )

    def __init__(self, calendar_endpoint: str, *, http: SafeHttpClient | None = None) -> None:
        self.calendar_endpoint = calendar_endpoint
        self.http = http or SafeHttpClient()

    def fetch(self, request: RetrievalRequest) -> ProviderBatch:
        started = datetime.now(UTC)
        calendar = self.http.get(self.calendar_endpoint, allowed_mime_prefixes=("text/html",))
        parser = _LinkParser()
        parser.feed(calendar.text)
        candidates: list[tuple[date, str]] = []
        for href in parser.links:
            url = urljoin(self.calendar_endpoint, href)
            match = self.STATEMENT.search(urlparse(url).path)
            if match:
                release_date = datetime.strptime(match.group(1), "%Y%m%d").date()
                if release_date <= request.as_of_utc.date():
                    candidates.append((release_date, url))
        if not candidates:
            raise ProviderFetchError("FED_LATEST_STATEMENT_MISSING", retryable=False)
        release_date, statement_url = max(candidates, key=lambda item: item[0])
        statement = self.http.get(statement_url, allowed_mime_prefixes=("text/html",))
        content, title = extract_official_text(statement.text)
        if len(content) < 100:
            raise ProviderFetchError("FED_STATEMENT_TOO_SHORT", retryable=False)
        eastern = ZoneInfo("America/New_York")
        published = datetime(
            release_date.year, release_date.month, release_date.day, 14, tzinfo=eastern
        ).astimezone(UTC)
        doc_id = f"federal_reserve:FOMC:{statement.sha256[:20]}"
        document = MacroDocument(
            doc_id=doc_id,
            source=self.provider_id,
            provider_id=self.provider_id,
            source_tier="A",
            event_type="FOMC",
            title=title or f"FOMC statement {release_date.isoformat()}",
            canonical_url=statement_url,
            content=content,
            published_at_utc=published,
            retrieved_at_utc=started,
            first_seen_at_utc=started,
            eligible_from_utc=started,
            content_sha256=statement.sha256,
            retrieval_method="HTML",
            official_primary=True,
            revision_status="INITIAL",
            independence_key=self.provider_id,
        )
        normalized_content = " ".join(content.split())
        target = self.TARGET.search(normalized_content)
        if target:
            normalized = f"{release_date.isoformat()}:{target.group(1)}:{target.group(2)}"
            value: dict[str, str] = {
                "target_lower": target.group(1),
                "target_upper": target.group(2),
            }
            text = (
                f"The Federal Reserve set the target range at {target.group(1)} to "
                f"{target.group(2)} percent on {release_date.isoformat()}."
            )
        else:
            normalized = f"{release_date.isoformat()}:{statement.sha256}"
            value = {"statement_date": release_date.isoformat()}
            text = f"The Federal Reserve published an FOMC statement on {release_date.isoformat()}."
        claim = MacroClaim(
            claim_id=f"claim:{statement.sha256[:32]}",
            slot="FOMC.latest_official_decision",
            event_type="FOMC",
            claim_type="latest_official_decision",
            normalized_value=normalized,
            display_text=text,
            value=value,
            reference_period=release_date.isoformat(),
            observed_at_utc=published,
            source_record_id=doc_id,
            provider_id=self.provider_id,
            source_tier="A",
            canonical_url=statement_url,
            published_at_utc=published,
            first_seen_at_utc=started,
            retrieved_at_utc=started,
            eligible_from_utc=started,
            content_sha256=statement.sha256,
            retrieval_method="HTML",
            official_primary=True,
            revision_status="INITIAL",
            independence_key=self.provider_id,
        )
        return ProviderBatch(
            provider_id=self.provider_id,
            source_tier=self.source_tier,
            required=self.required,
            started_at_utc=started,
            completed_at_utc=datetime.now(UTC),
            fetch_succeeded=True,
            documents=(document,),
            claims=(claim,),
            raw_response_sha256=(calendar.sha256, statement.sha256),
        )


@dataclass
class MacroRetrievalCoordinator:
    providers: tuple[MacroProvider, ...]
    optional_unavailable: dict[str, str] = field(default_factory=dict)

    def fetch(self, request: RetrievalRequest) -> tuple[ProviderBatch, ...]:
        batches: list[ProviderBatch] = []
        for provider in self.providers:
            started = datetime.now(UTC)
            try:
                batches.append(provider.fetch(request))
            except Exception as error:
                retryable = isinstance(error, ProviderFetchError) and error.retryable
                reason = (
                    error.reason_code
                    if isinstance(error, ProviderFetchError)
                    else type(error).__name__
                )
                batches.append(
                    ProviderBatch(
                        provider_id=provider.provider_id,
                        source_tier=provider.source_tier,
                        required=provider.required,
                        started_at_utc=started,
                        completed_at_utc=datetime.now(UTC),
                        fetch_succeeded=False,
                        error_type=type(error).__name__,
                        reason_code=reason,
                        retryable=retryable,
                    )
                )
        for provider_id, reason in sorted(self.optional_unavailable.items()):
            now = datetime.now(UTC)
            batches.append(
                ProviderBatch(
                    provider_id=provider_id,
                    source_tier="A",
                    required=False,
                    started_at_utc=now,
                    completed_at_utc=now,
                    fetch_succeeded=False,
                    reason_code=reason,
                    retryable=False,
                )
            )
        return tuple(batches)


def default_coordinator(
    config: dict[str, Any],
    *,
    disabled_sources: set[str] | None = None,
) -> MacroRetrievalCoordinator:
    endpoints = config["provider_endpoints"]
    disabled = disabled_sources or set()

    def selected(source_id: str, provider: MacroProvider) -> MacroProvider:
        if source_id in disabled:
            return DisabledProvider(provider_id=provider.provider_id, required=provider.required)
        return provider

    return MacroRetrievalCoordinator(
        providers=(
            selected(
                "federal_reserve",
                FedReleaseProvider("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
            ),
            selected("new_york_fed", NewYorkFedRatesProvider(str(endpoints["new_york_fed_effr"]))),
            selected(
                "us_treasury",
                TreasuryCurveProvider(str(endpoints["treasury_nominal"]), str(endpoints["treasury_real"])),
            ),
            selected(
                "bls",
                BlsReleaseProvider(str(endpoints["bls_api"]), registration_key=os.environ.get("BLS_API_KEY")),
            ),
            selected("bea", BeaReleaseProvider(str(endpoints["bea_rss"]))),
        ),
    )
