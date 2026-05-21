"""Wake a Streamlit Community Cloud app from GitHub Actions.

The Streamlit sleep screen is rendered by the platform before the app starts.
A simple HTTP request can receive only the sleep page and may not press the
"Yes, get this app back up!" button. This script uses a real Chromium browser
so the wake button is clicked when present, then waits until the actual app UI
or a non-sleep page is served.
"""

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
    "https://cococute88-opt-gorani-finance-v2-mmdtk57yfrbq4853tl3x4k.streamlit.app/",
)
MAX_WAIT_SECONDS = int(os.environ.get("STREAMLIT_WAKE_MAX_WAIT_SECONDS", "300"))
POLL_SECONDS = int(os.environ.get("STREAMLIT_WAKE_POLL_SECONDS", "10"))

SLEEP_TEXT = "This app has gone to sleep due to inactivity"
WAKE_BUTTON_TEXT = "Yes, get this app back up!"


def _write_debug(page) -> None:
    """Persist enough information to debug a failed scheduled wake run."""
    try:
        Path("wake_debug.html").write_text(page.content(), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - debug best effort
        print(f"Could not write wake_debug.html: {exc}")

    try:
        page.screenshot(path="wake_debug.png", full_page=True)
    except Exception as exc:  # pragma: no cover - debug best effort
        print(f"Could not write wake_debug.png: {exc}")


def _page_has_sleep_text(page) -> bool:
    try:
        return page.get_by_text(SLEEP_TEXT).count() > 0 or "Zzzz" in page.title()
    except PlaywrightError:
        return False


def _click_wake_button_if_present(page) -> bool:
    wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
    try:
        if wake_button.count() > 0:
            print("Streamlit sleep screen detected. Clicking wake button.")
            wake_button.first.click(timeout=15_000)
            return True
    except PlaywrightTimeoutError:
        print("Wake button was found but click timed out; will continue polling.")
    except PlaywrightError as exc:
        print(f"Wake button click failed; will continue polling: {exc}")
    return False


def main() -> int:
    print(f"Opening Streamlit app: {APP_URL}")
    deadline = time.monotonic() + MAX_WAIT_SECONDS

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1365, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
        )

        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

            clicked = _click_wake_button_if_present(page)
            if clicked:
                try:
                    page.wait_for_load_state("networkidle", timeout=60_000)
                except PlaywrightTimeoutError:
                    print("Network idle did not occur after wake click; continuing.")

            attempt = 1
            while time.monotonic() < deadline:
                title = page.title()
                current_url = page.url
                print(f"Attempt {attempt}: title={title!r}, url={current_url}")

                if not _page_has_sleep_text(page):
                    print("App is no longer showing the Streamlit sleep page.")
                    return 0

                _click_wake_button_if_present(page)
                time.sleep(POLL_SECONDS)
                try:
                    page.reload(wait_until="domcontentloaded", timeout=60_000)
                except PlaywrightTimeoutError:
                    print("Reload timed out; continuing until deadline.")
                attempt += 1

            print(f"Timed out after {MAX_WAIT_SECONDS}s waiting for app to wake.")
            _write_debug(page)
            return 1

        except Exception as exc:
            print(f"Unexpected wake failure: {exc}")
            _write_debug(page)
            return 1
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
