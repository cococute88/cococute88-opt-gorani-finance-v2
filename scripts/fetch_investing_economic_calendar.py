#!/usr/bin/env python3
"""Fetch U.S. high-importance economic calendar events from Investing.com.

The script is intended for GitHub Actions. It writes a metadata envelope to
``data/economic_calendar_us_high.json`` so the Streamlit app can distinguish a
successful empty calendar from request/parsing failures. Hard fetch/parsing
failures exit non-zero after preserving any existing valid successful payload.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, Tag

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "economic_calendar_us_high.json"
INVESTING_CALENDAR_URL = "https://www.investing.com/economic-calendar/"
INVESTING_ENDPOINT = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
KST = ZoneInfo("Asia/Seoul")
FETCH_DAYS = 35
REQUEST_WINDOW_DAYS = 14
TIMEOUT_SECONDS = 10


class PayloadCandidate(NamedTuple):
    name: str
    date_format: str
    timezone: str


class CandidateResult(NamedTuple):
    html: str
    candidate: PayloadCandidate
    events: list[dict[str, Any]]
    stats: dict[str, int]


PAYLOAD_CANDIDATES = (
    PayloadCandidate("candidate_1_ddmmyyyy_tz88", "%d/%m/%Y", "88"),
    PayloadCandidate("candidate_2_iso_tz88", "%Y-%m-%d", "88"),
    PayloadCandidate("candidate_3_ddmmyyyy_tz55", "%d/%m/%Y", "55"),
    PayloadCandidate("candidate_4_iso_tz55", "%Y-%m-%d", "55"),
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.investing.com",
    "Referer": INVESTING_CALENDAR_URL,
    "X-Requested-With": "XMLHttpRequest",
}

EVENT_NAME_MAPPINGS = [
    ("Philadelphia Fed Manufacturing Index", "필라델피아 연은 제조업활동지수"),
    ("Michigan Consumer Sentiment", "미시간대 소비자심리지수"),
    ("Fed Interest Rate Decision", "금리결정"),
    ("Interest Rate Decision", "금리결정"),
    ("FOMC", "금리결정"),
    ("Initial Jobless Claims", "신규 실업수당청구건수"),
    ("Crude Oil Inventories", "원유재고"),
    ("Core Retail Sales", "근원 소매판매"),
    ("Retail Sales", "소매판매"),
    ("Existing Home Sales", "기존주택판매"),
    ("New Home Sales", "신규주택판매"),
    ("Building Permits", "건축허가건수"),
    ("Housing Starts", "주택착공건수"),
    ("Non Farm Payrolls", "비농업고용지수"),
    ("Nonfarm Payrolls", "비농업고용지수"),
    ("Unemployment Rate", "실업률"),
    ("GDP Growth Rate", "GDP 성장률"),
    ("Core PCE Price Index", "근원 PCE 물가지수"),
    ("PCE Price Index", "PCE 물가지수"),
    ("Core CPI", "근원 소비자물가지수"),
    ("CPI", "소비자물가지수"),
    ("Core PPI", "근원 생산자물가지수"),
    ("PPI", "생산자물가지수"),
]


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def kst_today_range() -> tuple[date, date, datetime]:
    now = datetime.now(KST)
    start = now.date()
    return start, start + timedelta(days=FETCH_DAYS), now


def strip_report_period(raw_name: str) -> str:
    """Remove trailing release-period labels such as ``(May)`` while keeping ``(YoY)``/``(MoM)``."""
    text = clean_text(raw_name)
    period_pattern = re.compile(
        r"\s+\((?:"
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|"
        r"Q[1-4]|\d{4})"
        r"(?:\s+\d{4})?\)$",
        re.IGNORECASE,
    )
    last_text = None
    while last_text != text:
        last_text = text
        text = period_pattern.sub("", text).strip()
    return text


def translate_event_name(raw_name: str) -> str:
    raw_name = strip_report_period(raw_name)
    if not raw_name:
        return ""
    translated = raw_name
    for english, korean in EVENT_NAME_MAPPINGS:
        pattern = re.compile(re.escape(english), re.IGNORECASE)
        if pattern.search(translated):
            translated = pattern.sub(korean, translated, count=1)
            break
    return re.sub(r"\s+", " ", translated).strip()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def parse_investing_date(
    raw: str,
    current_date: date | None,
    timezone_id: str,
) -> tuple[str, str, datetime | None, str, str]:
    """Parse Investing.com row datetime and normalize it to KST.

    Investing.com returns ``data-event-datetime`` as a naive string. In the
    tested payload matrix, timezone id 88 is treated as already being Seoul/KST,
    while timezone id 55 is treated as UTC/GMT and converted to Asia/Seoul.
    The final two return values are compact raw/conversion strings for logging.
    """
    raw = clean_text(raw)
    candidates = [raw]
    if current_date and re.fullmatch(r"\d{1,2}:\d{2}", raw):
        candidates.insert(0, f"{current_date.isoformat()} {raw}")

    formats = (
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%b %d, %Y %H:%M",
        "%B %d, %Y %H:%M",
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                naive_dt = datetime.strptime(candidate, fmt)
                if timezone_id == "55":
                    source_dt = naive_dt.replace(tzinfo=ZoneInfo("UTC"))
                    kst_dt = source_dt.astimezone(KST)
                else:
                    source_dt = naive_dt.replace(tzinfo=KST)
                    kst_dt = source_dt
                return (
                    kst_dt.strftime("%Y-%m-%d"),
                    kst_dt.strftime("%H:%M"),
                    kst_dt,
                    f"{raw} ({source_dt.isoformat()})",
                    kst_dt.isoformat(),
                )
            except ValueError:
                continue

    time_match = re.search(r"(\d{1,2}:\d{2})", raw)
    if current_date and time_match:
        naive_dt = datetime.combine(current_date, time.fromisoformat(time_match.group(1)))
        if timezone_id == "55":
            source_dt = naive_dt.replace(tzinfo=ZoneInfo("UTC"))
            kst_dt = source_dt.astimezone(KST)
        else:
            source_dt = naive_dt.replace(tzinfo=KST)
            kst_dt = source_dt
        return (
            kst_dt.strftime("%Y-%m-%d"),
            kst_dt.strftime("%H:%M"),
            kst_dt,
            f"{raw} ({source_dt.isoformat()})",
            kst_dt.isoformat(),
        )
    return "", "", None, raw, ""

def parse_date_header(text: str, fallback_year: int) -> date | None:
    text = clean_text(text)
    text = re.sub(r"^[A-Za-z]+,\s*", "", text)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(text, fmt)
            year = parsed.year if "%Y" in fmt else fallback_year
            return date(year, parsed.month, parsed.day)
        except ValueError:
            continue
    return None


def importance_from_row(row: Tag) -> int:
    title_text = " ".join(
        clean_text(tag.get("title")) for tag in row.find_all(attrs={"title": True}) if clean_text(tag.get("title"))
    ).lower()
    if "high" in title_text:
        return 3

    full_bulls = 0
    for tag in row.find_all(True):
        classes = tag.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        class_text = " ".join(classes)
        if "grayFullBullishIcon" in class_text:
            continue
        if "fullBullishIcon" in class_text or "bull3" in class_text:
            full_bulls += 1
    return min(full_bulls, 3)


def cell_text(row: Tag, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        tag = row.select_one(selector)
        if tag:
            return clean_text(tag.get_text(" "))
    return ""


def extract_raw_name(row: Tag) -> str:
    name = cell_text(row, ("td.event a", "td.event", ".event"))
    if name:
        return name
    cols = [clean_text(td.get_text(" ")) for td in row.find_all("td")]
    return cols[3] if len(cols) > 3 else ""


def extract_event(
    row: Tag,
    current_date: date | None,
    start: date,
    end: date,
    updated_at: str,
    timezone_id: str,
    datetime_examples: list[tuple[str, str]],
) -> dict[str, Any] | None:
    importance = importance_from_row(row)
    if importance < 3:
        return None

    currency = cell_text(row, ("td.flagCur", ".flagCur", "td.left.flagCur")) or row.get("data-event-currency", "")
    if currency and currency.upper() != "USD":
        return None

    country_attr = clean_text(row.get("data-event-country"))
    if country_attr and country_attr.lower() not in {"united states", "usa", "us"}:
        return None

    raw_name = strip_report_period(extract_raw_name(row))
    if not raw_name:
        return None

    raw_datetime = clean_text(row.get("data-event-datetime")) or cell_text(row, ("td.first", "td.time", ".time"))
    event_date, event_time, sort_dt, raw_dt_log, kst_dt_log = parse_investing_date(raw_datetime, current_date, timezone_id)
    if not sort_dt:
        return None
    if raw_dt_log and kst_dt_log and not datetime_examples:
        datetime_examples.append((raw_dt_log, kst_dt_log))
    if sort_dt.date() < start or sort_dt.date() > end:
        return None

    return {
        "date": event_date,
        "time": event_time,
        "name": translate_event_name(raw_name),
        "raw_name": raw_name,
        "currency": "USD",
        "country": "United States",
        "importance": 3,
        "source": "investing",
        "updated_at": updated_at,
        "_sort": sort_dt.isoformat(),
    }


def response_preview(text: str, limit: int = 200) -> str:
    """Return a compact response-body preview without headers/cookies."""
    return clean_text(text)[:limit]


def is_cloudflare_challenge(text: str) -> bool:
    lowered = text.lower()
    challenge_markers = (
        "cloudflare",
        "cf-browser-verification",
        "cf-chl",
        "checking your browser",
        "just a moment",
    )
    return any(marker in lowered for marker in challenge_markers)


def base_payload(candidate: PayloadCandidate, start: date, end: date) -> dict[str, Any]:
    return {
        "country[]": ["5"],  # Investing.com country id for United States.
        "importance[]": ["3"],
        "dateFrom": start.strftime(candidate.date_format),
        "dateTo": end.strftime(candidate.date_format),
        "timeZone": candidate.timezone,
        "timeFilter": "timeRemain",
        "currentTab": "custom",
        "submitFilters": "1",
        "limit_from": "0",
    }


def html_counts(html: str) -> tuple[int, int]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr")
    raw_candidates = soup.select("tr.js-event-item, tr[id^='eventRowId_']")
    if not raw_candidates:
        raw_candidates = [
            row
            for row in rows
            if row.get("data-event-datetime") or str(row.get("id", "")).startswith("eventRowId_")
        ]
    return len(rows), len(raw_candidates)


def log_candidate_attempt(
    *,
    candidate: PayloadCandidate,
    start: date,
    end: date,
    response: requests.Response | None,
    payload: dict[str, Any],
    json_has_data: bool,
    html_row_count: int,
    raw_candidate_count: int,
    final_events_count: int | str,
    error: str | None = None,
) -> None:
    logging.info(
        (
            "Investing.com payload candidate result: name=%s date_format=%s timeZone=%s "
            "dateFrom=%s dateTo=%s status=%s length=%s content_type=%s preview=%r "
            "json_data=%s html_row_count=%s raw_event_row_candidates=%s final_events=%s error=%s"
        ),
        candidate.name,
        "DD/MM/YYYY" if candidate.date_format == "%d/%m/%Y" else "YYYY-MM-DD",
        candidate.timezone,
        payload["dateFrom"],
        payload["dateTo"],
        response.status_code if response is not None else "request_failed",
        len(response.text or "") if response is not None else 0,
        response.headers.get("Content-Type", "") if response is not None else "",
        response_preview(response.text or "") if response is not None else "",
        json_has_data,
        html_row_count,
        raw_candidate_count,
        final_events_count,
        error,
    )


def fetch_calendar_chunk(session: requests.Session, start: date, end: date, updated_at: str) -> CandidateResult:
    try:
        landing_response = session.get(INVESTING_CALENDAR_URL, timeout=TIMEOUT_SECONDS)
        logging.info(
            "Landing page response for chunk %s to %s: status=%s length=%s content_type=%s preview=%r",
            start,
            end,
            landing_response.status_code,
            len(landing_response.text or ""),
            landing_response.headers.get("Content-Type", ""),
            response_preview(landing_response.text or ""),
        )
        landing_response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Investing.com landing request failed: {exc}") from exc

    failures: list[str] = []
    for candidate in PAYLOAD_CANDIDATES:
        payload = base_payload(candidate, start, end)
        response: requests.Response | None = None
        json_has_data = False
        html_row_count = 0
        raw_candidate_count = 0
        final_events_count: int | str = 0
        error: str | None = None

        try:
            response = session.post(INVESTING_ENDPOINT, data=payload, timeout=TIMEOUT_SECONDS)
            text = response.text.strip()
            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
            elif is_cloudflare_challenge(text):
                error = "Cloudflare challenge detected"
            else:
                try:
                    decoded = response.json()
                except ValueError:
                    decoded = None
                    error = "response is not valid JSON"

                html = decoded.get("data") if isinstance(decoded, dict) else None
                json_has_data = isinstance(html, str) and bool(html.strip())
                if not json_has_data:
                    error = error or "JSON data field is missing or empty"
                else:
                    if is_cloudflare_challenge(html):
                        error = "Cloudflare challenge detected inside JSON data"
                    else:
                        html_row_count, raw_candidate_count = html_counts(html)
                        if html_row_count == 0 or raw_candidate_count == 0:
                            error = "JSON data HTML does not contain tr/js-event-item event rows"
                        else:
                            events, stats = parse_events(html, start, end, updated_at, candidate)
                            final_events_count = stats["final_events_count"]
                            log_candidate_attempt(
                                candidate=candidate,
                                start=start,
                                end=end,
                                response=response,
                                payload=payload,
                                json_has_data=json_has_data,
                                html_row_count=stats["html_row_count"],
                                raw_candidate_count=stats["raw_candidate_count"],
                                final_events_count=final_events_count,
                                error=None,
                            )
                            logging.info(
                                "Selected Investing.com payload candidate for %s to %s: name=%s date_format=%s timeZone=%s events=%d",
                                start,
                                end,
                                candidate.name,
                                "DD/MM/YYYY" if candidate.date_format == "%d/%m/%Y" else "YYYY-MM-DD",
                                candidate.timezone,
                                len(events),
                            )
                            return CandidateResult(html, candidate, events, stats)
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
        except Exception as exc:
            error = f"candidate parsing/validation failed: {exc}"

        log_candidate_attempt(
            candidate=candidate,
            start=start,
            end=end,
            response=response,
            payload=payload,
            json_has_data=json_has_data,
            html_row_count=html_row_count,
            raw_candidate_count=raw_candidate_count,
            final_events_count=final_events_count,
            error=error,
        )
        failures.append(f"{candidate.name}({payload['dateFrom']}..{payload['dateTo']}, tz={candidate.timezone}): {error}")

    raise RuntimeError(f"All Investing.com payload candidates failed for {start} to {end}: " + " | ".join(failures))

def row_currency(row: Tag) -> str:
    return cell_text(row, ("td.flagCur", ".flagCur", "td.left.flagCur")) or clean_text(row.get("data-event-currency"))


def row_country_is_us(row: Tag) -> bool:
    country_attr = clean_text(row.get("data-event-country"))
    return not country_attr or country_attr.lower() in {"united states", "usa", "us"}


def parse_events(
    html: str,
    start: date,
    end: date,
    updated_at: str,
    candidate: PayloadCandidate,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    soup = BeautifulSoup(html, "html.parser")
    events: list[dict[str, Any]] = []
    current_date: date | None = None
    html_row_count = 0
    raw_candidate_count = 0
    importance_pass_count = 0
    usd_pass_count = 0
    datetime_examples: list[tuple[str, str]] = []

    try:
        rows = soup.select("tr")
    except Exception as exc:
        raise RuntimeError("HTML parsing failed while selecting calendar rows") from exc

    html_row_count = len(rows)
    for row in rows:
        row_classes = row.get("class") or []
        row_text = clean_text(row.get_text(" "))
        if "theDay" in row_classes or row.select_one("td.theDay"):
            current_date = parse_date_header(row_text, start.year) or current_date
            continue
        is_candidate = "js-event-item" in row_classes or row.get("data-event-datetime") or row.get("id", "").startswith("eventRowId_")
        if not is_candidate:
            continue
        raw_candidate_count += 1
        if importance_from_row(row) >= 3:
            importance_pass_count += 1
        currency = row_currency(row)
        if (not currency or currency.upper() == "USD") and row_country_is_us(row):
            usd_pass_count += 1
        event = extract_event(row, current_date, start, end, updated_at, candidate.timezone, datetime_examples)
        if event:
            events.append(event)

    events.sort(key=lambda item: item.get("_sort", ""))
    cleaned = []
    seen = set()
    for event in events:
        dedupe_key = (event["date"], event["time"], event["raw_name"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        event.pop("_sort", None)
        cleaned.append(event)

    logging.info(
        "Parsing stats for %s to %s using %s: html_rows=%d raw_event_candidates=%d importance_3_pass=%d usd_filter_pass=%d final_events=%d",
        start,
        end,
        candidate.name,
        html_row_count,
        raw_candidate_count,
        importance_pass_count,
        usd_pass_count,
        len(cleaned),
    )
    if datetime_examples:
        raw_example, kst_example = datetime_examples[0]
        logging.info(
            "Datetime conversion example for %s to %s using %s timezone=%s: raw=%r converted_kst=%r",
            start,
            end,
            candidate.name,
            candidate.timezone,
            raw_example,
            kst_example,
        )

    stats = {
        "html_row_count": html_row_count,
        "raw_candidate_count": raw_candidate_count,
        "importance_pass_count": importance_pass_count,
        "usd_pass_count": usd_pass_count,
        "final_events_count": len(cleaned),
    }

    if html_row_count == 0:
        raise RuntimeError("HTML parsing found zero table rows")
    if raw_candidate_count == 0:
        logging.warning("No raw event row candidates found for %s to %s", start, end)
    if not cleaned:
        logging.warning(
            "No final events for %s to %s after filters; candidates=%d importance_3_pass=%d usd_filter_pass=%d",
            start,
            end,
            raw_candidate_count,
            importance_pass_count,
            usd_pass_count,
        )
    return cleaned, stats


def iter_request_ranges(start: date, end: date) -> list[tuple[date, date]]:
    """Split the 35-day target range into smaller inclusive Investing.com requests."""
    ranges: list[tuple[date, date]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=REQUEST_WINDOW_DAYS), end)
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return ranges


def fetch_events(start: date, end: date, updated_at: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for chunk_start, chunk_end in iter_request_ranges(start, end):
        logging.info("Requesting Investing.com chunk from %s to %s", chunk_start, chunk_end)
        result = fetch_calendar_chunk(session, chunk_start, chunk_end, updated_at)
        events.extend(result.events)

    events.sort(key=lambda item: (item["date"], item["time"], item["raw_name"]))
    deduped: list[dict[str, Any]] = []
    seen = set()
    for event in events:
        dedupe_key = (event["date"], event["time"], event["raw_name"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(event)
    return deduped


def build_payload(status: str, updated_at: str | None, events: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at": updated_at,
        "source": "investing",
        "events": events,
        "error": error,
    }
    return payload


def write_json(payload: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(OUTPUT_PATH)


def load_existing_payload() -> Any:
    if not OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Existing economic calendar JSON is invalid and cannot be preserved as valid data: %s", exc)
        return None


def has_existing_successful_events(payload: Any) -> bool:
    if isinstance(payload, list):
        return len([item for item in payload if isinstance(item, dict)]) > 0
    if isinstance(payload, dict):
        events = payload.get("events")
        return payload.get("status") == "ok" and isinstance(events, list) and len(events) > 0
    return False


def preserve_existing_or_write_failure(reason: str, updated_at: str) -> None:
    logging.error("Economic calendar fetch failed: %s", reason)
    existing = load_existing_payload()
    if has_existing_successful_events(existing):
        existing_events = existing if isinstance(existing, list) else existing.get("events", [])
        existing_updated_at = existing[0].get("updated_at") if isinstance(existing, list) and existing else existing.get("updated_at")
        logging.warning(
            "Writing fetch_failed status while preserving %d existing events from last successful JSON",
            len(existing_events),
        )
        write_json(build_payload("fetch_failed", existing_updated_at or updated_at, existing_events, error=reason[:500]))
        return
    logging.warning("Writing fetch_failed status JSON to %s", OUTPUT_PATH)
    write_json(build_payload("fetch_failed", updated_at, [], error=reason[:500]))


def main() -> int:
    setup_logging()
    start, end, now = kst_today_range()
    updated_at = now.isoformat(timespec="seconds")
    logging.info("Fetching Investing.com U.S. high-importance calendar from %s to %s", start, end)

    try:
        events = fetch_events(start, end, updated_at)
        status = "ok" if events else "empty"
        if not events:
            logging.warning(
                "Investing.com collection succeeded but final events count is 0 after date/importance/USD filters for %s to %s",
                start,
                end,
            )
        write_json(build_payload(status, updated_at, events))
        logging.info("Wrote status=%s events=%d to %s", status, len(events), OUTPUT_PATH)
        return 0
    except Exception as exc:
        preserve_existing_or_write_failure(str(exc), updated_at)
        logging.error("Exiting with code 1 so GitHub Actions does not hide the fetch/parsing failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
