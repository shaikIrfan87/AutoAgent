import math
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import trafilatura
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

logger = logging.getLogger("HybridScraper")

class IngressSaliencyGate:
    """Invariant 1: Epistemic Gating and Adversarial Token Pre-Filtering."""
    
    @staticmethod
    def compute_entropy(text: str) -> float:
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        return -sum(p * math.log2(p) for p in prob)

    @staticmethod
    def compute_oov_anomaly_ratio(text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        anomalous = sum(1 for w in words if len(w) > 35 or sum(c.isdigit() for c in w) / len(w) > 0.4)
        return anomalous / len(words)

    @classmethod
    def filter_incoming_content(cls, raw_content: str) -> bool:
        """Enforces H(X) >= 2.0 and flags adversarial payloads."""
        if cls.compute_entropy(raw_content) < 2.0:
            logger.warning("Ingress content dropped: Low Shannon entropy (spam/boilerplate).")
            return False
        if cls.compute_oov_anomaly_ratio(raw_content) > 0.30:
            logger.warning("Ingress content dropped: High adversarial fragment/OOV ratio.")
            return False
        return True


class HybridWebScraper:
    """Perception Gateway combining sub-millisecond static parsing with dynamic fallback."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.gate = IngressSaliencyGate()

    def scrape_static(self, url: str) -> Dict[str, Any]:
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                # Stdlib fallback when trafilatura fetch fails
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    downloaded = resp.read().decode("utf-8", errors="ignore")

            if not downloaded:
                return {"status": "error", "error": "Empty response", "url": url}

            text = trafilatura.extract(
                downloaded, 
                include_links=False, 
                include_images=False, 
                output_format="txt"
            ) or ""

            soup = BeautifulSoup(downloaded, "html.parser")
            if not text:
                text = soup.get_text(separator=" ", strip=True)

            code_blocks = [
                tag.get_text().strip() 
                for tag in soup.find_all(["pre", "code"]) 
                if len(tag.get_text().strip()) > 30
            ]

            return {
                "status": "success",
                "engine": "trafilatura",
                "url": url,
                "domain": urlparse(url).netloc,
                "content": text.strip(),
                "code_blocks": code_blocks[:10],
                "char_count": len(text)
            }
        except Exception as ex:
            return {"status": "error", "error": str(ex), "url": url}

    async def scrape_dynamic(self, url: str) -> Dict[str, Any]:
        if not CRAWL4AI_AVAILABLE:
            return self.scrape_static(url)

        try:
            browser_conf = BrowserConfig(headless=self.headless)
            run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

            async with AsyncWebCrawler(config=browser_conf) as crawler:
                res = await crawler.arun(url=url, config=run_conf)
                content = res.markdown or res.extracted_content or ""
                
                soup = BeautifulSoup(res.html or "", "html.parser")
                code_blocks = [
                    tag.get_text().strip() 
                    for tag in soup.find_all(["pre", "code"]) 
                    if len(tag.get_text().strip()) > 30
                ]

                return {
                    "status": "success",
                    "engine": "crawl4ai",
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "content": content.strip(),
                    "code_blocks": code_blocks[:10],
                    "char_count": len(content)
                }
        except Exception as ex:
            return {"status": "error", "error": str(ex), "url": url}

    def fetch(self, url: str) -> Dict[str, Any]:
        """Auto-evaluating router: attempts fast-path first, falls back to dynamic."""
        result = self.scrape_static(url)
        # Fallback to headless JS rendering if page is empty or an SPA mount shell
        if result.get("status") == "error" or result.get("char_count", 0) < 200:
            import asyncio
            result = asyncio.run(self.scrape_dynamic(url))

        # Apply Invariant 1 Gating
        if result.get("status") == "success":
            passed = self.gate.filter_incoming_content(result.get("content", ""))
            result["saliency_passed"] = passed
        return result
