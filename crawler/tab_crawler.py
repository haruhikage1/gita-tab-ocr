import asyncio
import os
import time

import aiohttp

from gtrs.simple_logging import eprint


class TabCrawler:
    def __init__(
        self,
        output_dir: str,
        min_interval: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 60.0,
    ) -> None:
        self.output_dir = output_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_request_time = 0.0
        self._downloaded = 0
        self._failed = 0

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _download_one(
        self, session: aiohttp.ClientSession, url: str, filepath: str
    ) -> bool:
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429 or resp.status == 403:
                        eprint(f"Rate limited ({resp.status}), waiting {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                        continue
                    if resp.status != 200:
                        eprint(f"HTTP {resp.status} for {url}")
                        return False

                    data = await resp.read()
                    if len(data) < 1024:
                        eprint(f"Download too small ({len(data)} bytes), skipping")
                        return False

                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(data)

                    self._downloaded += 1
                    return True

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                eprint(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        self._failed += 1
        return False

    async def crawl_urls(self, urls: list[str]) -> dict[str, int]:
        os.makedirs(self.output_dir, exist_ok=True)
        connector = aiohttp.TCPConnector(limit=3)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for url in urls:
                filename = url.split("/")[-1].split("?")[0]
                if not filename.endswith((".png", ".jpg", ".jpeg")):
                    filename += ".png"
                filepath = os.path.join(self.output_dir, filename)
                tasks.append(self._download_one(session, url, filepath))

            await asyncio.gather(*tasks)

        stats = {"downloaded": self._downloaded, "failed": self._failed}
        eprint(f"Crawl complete: {stats}")
        return stats

    async def crawl_search(
        self, keywords: list[str], max_pages: int = 5
    ) -> dict[str, int]:
        eprint(f"Search crawling for keywords: {keywords} (stub - implement per-site logic)")
        return {"downloaded": 0, "failed": 0}