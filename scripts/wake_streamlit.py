"""Wake and validate a Streamlit Community Cloud app with a real browser."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://cococute88-opt-gorani-finance-v2-mmdtk57yfrbq4853tl3x4k.streamlit.app",
)
EXPECTED_READY_TEXT = (os.environ.get("EXPECTED_READY_TEXT") or "").strip()
MAX_WAIT_SECONDS = int(os.environ.get("MAX_WAIT_SECONDS", "900"))
SCREENSHOT_PATH = os.environ.get("SCREENSHOT_PATH", "artifacts/streamlit-wakeup.png")
HTML_PATH = os.environ.get("HTML_PATH", "artifacts/streamlit-wakeup.html")

POLL_SECONDS = 15

SLEEP_SIGNALS = [
    "your app is in the oven",
    "zzzz",
    "this app has gone to sleep",
    "gone to sleep",
    "yes, get this app back up",
    "get this app back up",
]

WAKE_BUTTON_CANDIDATES = [
    "Yes, get this app back up!",
    "Get this app back up",
    "Wake",
]


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _snapshot(page) -> tuple[str, str, str]:
    title = page.title()
    url = page.url
    body = (page.inner_text("body") or "")[:1200]
    return title, url, body


def _contains_sleep_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in SLEEP_SIGNALS)


def _write_debug_artifacts(page) -> None:
    _ensure_parent(SCREENSHOT_PATH)
    _ensure_parent(HTML_PATH)
    Path(HTML_PATH).write_text(page.content(), encoding="utf-8")
    page.screenshot(path=SCREENSHOT_PATH, full_page=True)


def _click_wake_button_if_present(page) -> bool:
    for label in WAKE_BUTTON_CANDIDATES:
        button = page.get_by_role("button", name=label)
        try:
            if button.count() > 0:
                print(f"[B] Sleep page detected. Clicking wake button: {label!r}")
                button.first.click(timeout=15_000)
                return True
        except PlaywrightError as exc:
            print(f"[B] Wake button interaction failed for {label!r}: {exc}")

    try:
        generic_wake = page.locator("button", has_text="Wake")
        if generic_wake.count() > 0:
            print("[B] Clicking generic 'Wake' button.")
            generic_wake.first.click(timeout=10_000)
            return True
    except PlaywrightError as exc:
        print(f"[B] Generic wake click failed: {exc}")
    return False


def main() -> int:
    print("[INFO] Wakeup script started")
    print("[A] If this workflow never runs, check default branch, workflow path, and schedule trigger.")
    print(f"[INFO] Target URL: {APP_URL}")
    print(f"[INFO] MAX_WAIT_SECONDS={MAX_WAIT_SECONDS}, EXPECTED_READY_TEXT={EXPECTED_READY_TEXT!r}")

    deadline = time.monotonic() + MAX_WAIT_SECONDS

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
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
            attempt = 1

            while time.monotonic() < deadline:
                try:
                    body_text = page.inner_text("body")
                except PlaywrightError:
                    body_text = ""

                title, current_url, body_preview = _snapshot(page)
                sleep_visible = _contains_sleep_signal(body_text) or _contains_sleep_signal(title)
                ready_visible = bool(EXPECTED_READY_TEXT and EXPECTED_READY_TEXT in body_text)

                print(f"[INFO] Attempt {attempt} title={title!r} url={current_url}")

                if not sleep_visible:
                    if EXPECTED_READY_TEXT and not ready_visible:
                        print("[D] App page loaded but expected ready marker not found yet.")
                    else:
                        print("Streamlit app wakeup success")
                        _write_debug_artifacts(page)
                        return 0

                clicked = _click_wake_button_if_present(page)
                if sleep_visible and not clicked:
                    print("[B] Sleep/oven screen still visible; retrying with reload.")

                time.sleep(POLL_SECONDS)
                try:
                    page.reload(wait_until="domcontentloaded", timeout=90_000)
                except PlaywrightTimeoutError:
                    print("[C] Reload timed out (possible slow app boot / dependency install / secrets issue).")

                attempt += 1

            print("[C] App did not become ready before timeout; likely app boot problem or Streamlit Cloud delay.")
            title, current_url, body_preview = _snapshot(page)
            print(f"[DEBUG] Final URL: {current_url}")
            print(f"[DEBUG] Final title: {title}")
            print(f"[DEBUG] Body preview:\n{body_preview}")
            _write_debug_artifacts(page)
            return 1

        except Exception as exc:
            print(f"[C] Unexpected wakeup failure: {exc}")
            try:
                title, current_url, body_preview = _snapshot(page)
                print(f"[DEBUG] Final URL: {current_url}")
                print(f"[DEBUG] Final title: {title}")
                print(f"[DEBUG] Body preview:\n{body_preview}")
            except Exception:
                pass
            _write_debug_artifacts(page)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
