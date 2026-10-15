"""Polite, cached HTTP client for official CMS sources.

* Minimum interval between requests (default 1.5 s).
* Retries with exponential backoff on timeouts, 429 and 5xx (honours Retry-After).
* Disk cache keyed by URL so re-runs are incremental and do not re-hit CMS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from careimpact.config import Paths, settings

log = logging.getLogger(__name__)


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    from_cache: bool
    fetched_at: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200


class PoliteClient:
    def __init__(self, cache_dir: Path | None = None, min_interval: float | None = None):
        cfg = settings()
        self.cache_dir = cache_dir or Paths.http_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = cfg["http_min_interval_s"] if min_interval is None else min_interval
        self.max_retries = cfg["http_max_retries"]
        self._last = 0.0
        self._client = httpx.Client(
            timeout=cfg["http_timeout_s"], follow_redirects=True,
            headers={"User-Agent": cfg["user_agent"], "Accept-Language": "en-US"},
        )

    def _cache_paths(self, url: str, headers: dict | None) -> tuple[Path, Path]:
        key = hashlib.sha256((url + json.dumps(headers or {}, sort_keys=True)).encode()).hexdigest()[:32]
        return self.cache_dir / f"{key}.body", self.cache_dir / f"{key}.json"

    def get(self, url: str, headers: dict | None = None, use_cache: bool = True, cache_key_headers: bool = False) -> FetchResult:
        body_p, meta_p = self._cache_paths(url, headers if cache_key_headers else None)
        if use_cache and body_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return FetchResult(url, meta["final_url"], meta["status"], body_p.read_text(encoding="utf-8"), True, meta["fetched_at"])

        err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._client.get(url, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                err = f"{type(exc).__name__}: {exc}"
                log.warning("fetch %s failed (%s), attempt %d", url, err, attempt + 1)
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt * 3)
                err = f"HTTP {r.status_code}"
                log.warning("fetch %s -> %s, backing off %.0fs", url, r.status_code, delay)
                time.sleep(delay)
                continue
            fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            r.encoding = r.encoding or "utf-8"
            text = r.text
            if r.status_code == 200:
                body_p.write_text(text, encoding="utf-8")
                meta_p.write_text(json.dumps({"url": url, "final_url": str(r.url), "status": r.status_code,
                                              "fetched_at": fetched_at}), encoding="utf-8")
            return FetchResult(url, str(r.url), r.status_code, text, False, fetched_at,
                               None if r.status_code == 200 else f"HTTP {r.status_code}")
        return FetchResult(url, url, 0, "", False, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), err or "failed")

    def get_json(self, url: str, headers: dict | None = None, use_cache: bool = True) -> dict:
        res = self.get(url, headers=headers, use_cache=use_cache)
        if not res.ok:
            raise RuntimeError(f"GET {url} failed: {res.error}")
        return json.loads(res.text)
