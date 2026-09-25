import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


class VirtualBrowserController:
    """Runs browser actions in a disposable sandbox using an isolated user profile clone."""

    def __init__(self):
        self.temp_profile_dir: Optional[str] = None
        self.playwright = None
        self.context = None

    def launch_virtual(self):
        # 1. Source real profile
        real_profile = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
        self.temp_profile_dir = tempfile.mkdtemp(prefix="autoagent_chrome_clone_")

        # 2. Copy only default cookies and session states, not real history/passwords
        src_default = real_profile / "Default"
        dest_default = Path(self.temp_profile_dir) / "Default"
        dest_default.mkdir(parents=True, exist_ok=True)

        for file in ["Network", "Cookies", "Session Storage"]:
            target = src_default / file
            if target.exists():
                try:
                    if target.is_dir():
                        shutil.copytree(target, dest_default / file, dirs_exist_ok=True)
                    else:
                        shutil.copyfile(target, dest_default / file)
                except Exception:
                    pass

        # 3. Launch isolated Playwright context
        try:
            from playwright.sync_api import sync_playwright

            self.playwright = sync_playwright().start()
            try:
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.temp_profile_dir,
                    channel="chrome",
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception:
                self.context = self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.temp_profile_dir,
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
        except Exception:
            self.context = None

    def scrape_untrusted_page(self, url: str) -> str:
        """Navigates and extracts raw DOM text inside the virtual browser."""
        if "127.0.0.1" in url or "localhost" in url:
            try:
                import urllib.request
                import re
                req = urllib.request.Request(url, headers={"User-Agent": "AutoAgent/1.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    return re.sub(r"<[^>]+>", " ", html)[:15000]
            except Exception:
                pass

        if not self.context:
            self.launch_virtual()
        if not self.context:
            return ""
        page = self.context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=10000)
            return page.locator("body").inner_text()[:15000]
        except Exception:
            try:
                import urllib.request
                import re
                req = urllib.request.Request(url, headers={"User-Agent": "AutoAgent/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    return re.sub(r"<[^>]+>", " ", html)[:15000]
            except Exception:
                return ""
        finally:
            try:
                page.close()
            except Exception:
                pass

    def close(self):
        if self.context:
            try:
                self.context.close()
            except Exception:
                pass
            self.context = None
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
        if self.temp_profile_dir and Path(self.temp_profile_dir).exists():
            shutil.rmtree(self.temp_profile_dir, ignore_errors=True)
            self.temp_profile_dir = None
