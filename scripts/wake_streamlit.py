"""Wake and validate a Streamlit Community Cloud app with a real browser."""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://cococute88-opt-gorani-finance-v2-mmdtk57yfrbq4853tl3x4k.streamlit.app/",
)
EXPECTED_READY_TEXT = os.environ.get("EXPECTED_READY_TEXT", "GORANI_FINANCE_APP_READY").strip()
MAX_WAIT_SECONDS = int(os.environ.get("MAX_WAIT_SECONDS", "1200"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "20"))
SCREENSHOT_PATH = os.environ.get("SCREENSHOT_PATH", "artifacts/streamlit-wakeup.png")
HTML_PATH = os.environ.get("HTML_PATH", "artifacts/streamlit-wakeup.html")

BODY_PREVIEW_CHARS = 1800
POST_WAKE_WAIT_SECONDS = 45
OVEN_REFRESH_SECONDS = 120
NORMAL_REFRESH_SECONDS = 75

WAKE_PATTERN = re.compile(
    r"yes,?\s*get\s*this\s*app\s*back\s*up!?|"
    r"get\s*this\s*app\s*back\s*up|"
    r"wake\s*up|"
    r"\bwake\b",
    re.IGNORECASE,
)

SLEEP_PATTERN = re.compile(
    r"this\s*app\s*has\s*gone\s*to\s*sleep|"
    r"this\s*app\s*is\s*asleep|"
    r"zzzz|"
    r"get\s*this\s*app\s*back\s*up|"
    r"yes,?\s*get\s*this\s*app\s*back\s*up!?",
    re.IGNORECASE,
)
OVEN_PATTERN = re.compile(r"your\s+app\s+is\s+in\s+the\s+oven", re.IGNORECASE)
ERROR_PATTERN = re.compile(
    r"app\s+error|uncaught\s+app\s+exception|traceback|module\s+not\s+found|"
    r"failed\s+to\s+load|connection\s+error|internal\s+server\s+error",
    re.IGNORECASE,
)
STREAMLIT_UI_PATTERN = re.compile(
    r"gorani|google|로그인|login|portfolio|asset|finance|자산|배당",
    re.IGNORECASE,
)


@dataclass
class PageState:
    title: str
    url: str
    body: str

    @property
    def preview(self) -> str:
        return self.body[:BODY_PREVIEW_CHARS]

    @property
    def combined(self) -> str:
        return f"{self.title}\n{self.url}\n{self.body}"

    @property
    def has_ready_marker(self) -> bool:
        return bool(EXPECTED_READY_TEXT and EXPECTED_READY_TEXT in self.body)

    @property
    def has_sleep_signal(self) -> bool:
        return bool(SLEEP_PATTERN.search(self.combined))

    @property
    def has_oven_signal(self) -> bool:
        return bool(OVEN_PATTERN.search(self.combined))

    @property
    def has_error_signal(self) -> bool:
        return bool(ERROR_PATTERN.search(self.combined))

    @property
    def is_success_candidate(self) -> bool:
        return (
            bool(STREAMLIT_UI_PATTERN.search(self.combined))
            and not self.has_sleep_signal
            and not self.has_oven_signal
            and not self.has_error_signal
        )


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_body_text(page) -> str:
    try:
        return page.locator("body").inner_text(timeout=10_000) or ""
    except PlaywrightError as exc:
        print(f"[WARN] Could not read body text yet: {exc}")
        return ""


def _snapshot(page) -> PageState:
    try:
        title = page.title()
    except PlaywrightError as exc:
        title = f"<title unavailable: {exc}>"
    return PageState(title=title, url=page.url, body=_read_body_text(page))


def _write_debug_artifacts(page) -> None:
    _ensure_parent(SCREENSHOT_PATH)
    _ensure_parent(HTML_PATH)

    try:
        Path(HTML_PATH).write_text(page.content(), encoding="utf-8")
        print(f"[ARTIFACT] HTML saved: {HTML_PATH}")
    except PlaywrightError as exc:
        Path(HTML_PATH).write_text(f"<!-- page.content() failed: {exc} -->", encoding="utf-8")
        print(f"[ARTIFACT] HTML fallback saved after content failure: {HTML_PATH}")

    try:
        page.screenshot(path=SCREENSHOT_PATH, full_page=True, timeout=30_000)
        print(f"[ARTIFACT] Screenshot saved: {SCREENSHOT_PATH}")
    except PlaywrightError as exc:
        print(f"[ARTIFACT] Screenshot save failed: {exc}")


def _log_visible_buttons(page) -> None:
    try:
        buttons = page.locator("button")
        count = buttons.count()
    except PlaywrightError as exc:
        print(f"[DEBUG] Could not list buttons: {exc}")
        return

    visible_texts: list[str] = []
    for index in range(min(count, 20)):
        button = buttons.nth(index)
        try:
            if button.is_visible(timeout=1_000):
                visible_texts.append((button.inner_text(timeout=1_000) or "").strip())
        except PlaywrightError:
            continue

    if visible_texts:
        print(f"[DEBUG] Visible button texts: {visible_texts}")
    else:
        print("[DEBUG] No visible buttons found.")


def _click_locator(locator, description: str) -> bool:
    try:
        count = locator.count()
    except PlaywrightError as exc:
        print(f"[WAKE] Could not count candidates via {description}: {exc}")
        return False

    for index in range(min(count, 10)):
        candidate = locator.nth(index)
        try:
            if not candidate.is_visible(timeout=3_000):
                continue
            print(f"[WAKE] Clicking wake control via {description} (candidate {index + 1}/{count}).")
            candidate.click(timeout=15_000)
            return True
        except PlaywrightError as exc:
            print(f"[WAKE] Wake control attempt failed via {description} candidate {index + 1}: {exc}")
    return False


def _click_wake_button_if_present(page) -> bool:
    _log_visible_buttons(page)

    strategies = [
        (page.get_by_role("button", name=WAKE_PATTERN), "get_by_role(button, wake regex)"),
        (page.get_by_text(WAKE_PATTERN), "get_by_text(wake regex)"),
        (page.locator("button").filter(has_text=WAKE_PATTERN), "button filtered by wake regex"),
    ]

    for locator, description in strategies:
        if _click_locator(locator, description):
            return True
    return False


def _sleep_until(deadline: float, seconds: int, reason: str) -> None:
    remaining = max(0, deadline - time.monotonic())
    delay = min(seconds, int(remaining))
    if delay > 0:
        print(f"[WAIT] {reason}; sleeping {delay}s.")
        time.sleep(delay)


def _reload(page) -> None:
    try:
        print("[NAV] Reloading page with domcontentloaded wait.")
        page.reload(wait_until="domcontentloaded", timeout=90_000)
    except PlaywrightTimeoutError:
        print("[WARN] Reload timed out; continuing to poll current document.")
    except PlaywrightError as exc:
        print(f"[WARN] Reload failed; continuing to poll current document: {exc}")


def _classify_final_failure(state: PageState) -> str:
    if state.has_oven_signal:
        return "Streamlit Cloud oven/build screen stayed visible until timeout."
    if state.has_sleep_signal:
        return "Streamlit Cloud sleep screen or wake button stayed visible until timeout."
    if state.has_error_signal:
        return "The app appears to show an application/runtime error."
    if EXPECTED_READY_TEXT and not state.has_ready_marker:
        return "The page loaded but the expected ready marker was not found."
    return "Unknown timeout while waiting for the app to become ready."


def main() -> int:
    print("[INFO] Wakeup script started")
    print(f"[INFO] Target URL: {APP_URL}")
    print(f"[INFO] EXPECTED_READY_TEXT={EXPECTED_READY_TEXT!r}")
    print(f"[INFO] MAX_WAIT_SECONDS={MAX_WAIT_SECONDS}, POLL_SECONDS={POLL_SECONDS}")
    print(f"[INFO] SCREENSHOT_PATH={SCREENSHOT_PATH}, HTML_PATH={HTML_PATH}")

    deadline = time.monotonic() + MAX_WAIT_SECONDS
    last_reload_at = 0.0
    fallback_success_seen = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1366, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        )

        try:
            print("[NAV] Opening app with domcontentloaded wait.")
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
            attempt = 1

            while time.monotonic() < deadline:
                state = _snapshot(page)
                remaining = int(deadline - time.monotonic())
                print("-" * 80)
                print(f"[ATTEMPT {attempt}] remaining={remaining}s")
                print(f"[ATTEMPT {attempt}] title={state.title!r}")
                print(f"[ATTEMPT {attempt}] url={state.url}")
                print(f"[ATTEMPT {attempt}] body preview:\n{state.preview}")
                print(
                    f"[ATTEMPT {attempt}] flags "
                    f"ready={state.has_ready_marker} sleep={state.has_sleep_signal} "
                    f"oven={state.has_oven_signal} error={state.has_error_signal} "
                    f"candidate={state.is_success_candidate}"
                )

                if state.has_ready_marker:
                    print("[SUCCESS] Expected ready marker found in page body.")
                    _write_debug_artifacts(page)
                    return 0

                if state.has_sleep_signal:
                    clicked = _click_wake_button_if_present(page)
                    if clicked:
                        _sleep_until(deadline, POST_WAKE_WAIT_SECONDS, "wake button clicked; allowing Streamlit to boot")
                    else:
                        print("[WAKE] Sleep signal found, but no clickable wake control matched.")
                    fallback_success_seen = 0
                elif state.has_oven_signal:
                    print("[INFO] Streamlit oven/build screen detected; waiting without repeated wake clicks.")
                    _sleep_until(deadline, max(POLL_SECONDS, 45), "oven/build screen is still preparing the app")
                    fallback_success_seen = 0
                elif state.has_error_signal:
                    print("[WARN] Possible app error text detected; continuing until timeout for artifact capture.")
                    _sleep_until(deadline, POLL_SECONDS, "possible app error observed")
                    fallback_success_seen = 0
                elif state.is_success_candidate:
                    fallback_success_seen += 1
                    print(
                        "[WARN] Ready marker is absent, but a non-sleep Streamlit/app UI is visible "
                        f"({fallback_success_seen}/2 confirmation)."
                    )
                    if fallback_success_seen >= 2:
                        print("[SUCCESS] Treating stable rendered app UI as a fallback success candidate.")
                        _write_debug_artifacts(page)
                        return 0
                    _sleep_until(deadline, POLL_SECONDS, "confirming fallback success candidate")
                else:
                    print("[INFO] Page not ready yet and no known sleep/oven/error state identified.")
                    _sleep_until(deadline, POLL_SECONDS, "polling page body")
                    fallback_success_seen = 0

                now = time.monotonic()
                refresh_interval = OVEN_REFRESH_SECONDS if state.has_oven_signal else NORMAL_REFRESH_SECONDS
                if now - last_reload_at >= refresh_interval and time.monotonic() < deadline:
                    last_reload_at = now
                    _reload(page)

                attempt += 1

            final_state = _snapshot(page)
            print("-" * 80)
            print(f"[FAILURE] {_classify_final_failure(final_state)}")
            print(f"[DEBUG] Final URL: {final_state.url}")
            print(f"[DEBUG] Final title: {final_state.title}")
            print(f"[DEBUG] Final body preview:\n{final_state.preview}")
            _write_debug_artifacts(page)
            return 1

        except Exception as exc:
            print(f"[FAILURE] Unexpected wakeup failure: {exc}")
            try:
                final_state = _snapshot(page)
                print(f"[DEBUG] Final URL: {final_state.url}")
                print(f"[DEBUG] Final title: {final_state.title}")
                print(f"[DEBUG] Final body preview:\n{final_state.preview}")
            except Exception as snapshot_exc:
                print(f"[DEBUG] Could not capture final page state: {snapshot_exc}")
            _write_debug_artifacts(page)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
