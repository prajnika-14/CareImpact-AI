"""CMS Coverage API client (https://api.coverage.cms.gov).

Verified behaviour (probed 2026-10):
* /v1/reports/local-coverage-final-lcds/ lists current final LCDs with
  document_id, document_version, title, contractor, effective/retirement dates
  and the MCD URL. No token required.
* /v1/data/lcd/?lcdid=X&ver=Y returns data only for the *current* version; older
  versions return an empty data array. Historical text therefore comes from the
  MCD website (see mcd_pages.py).
* /v1/data/* endpoints need a Bearer token from /v1/metadata/license-agreement,
  which implies accepting AMA/ADA/AHA license terms. Gated behind the
  ``accept_cms_api_license`` setting.
"""
from __future__ import annotations

import html
import re
from typing import Iterator

from careimpact.config import settings
from careimpact.ingestion.http import PoliteClient

API = "https://api.coverage.cms.gov/v1"


class CoverageAPI:
    def __init__(self, client: PoliteClient | None = None):
        self.http = client or PoliteClient()
        self._token: str | None = None

    def _paged(self, path: str, params: str = "", use_cache: bool = True) -> Iterator[dict]:
        token = ""
        while True:
            sep = "&" if params else ""
            url = f"{API}{path}?{params}{sep}next_token={token}" if token else f"{API}{path}{'?' + params if params else ''}"
            data = self.http.get_json(url, use_cache=use_cache)
            yield from data.get("data", [])
            token = data.get("meta", {}).get("next_token") or ""
            if not token:
                break

    def final_lcds(self, use_cache: bool = True) -> list[dict]:
        return list(self._paged("/reports/local-coverage-final-lcds/", use_cache=use_cache))

    def articles(self, use_cache: bool = True) -> list[dict]:
        return list(self._paged("/reports/local-coverage-articles/", use_cache=use_cache))

    def token(self) -> str:
        if not settings()["accept_cms_api_license"]:
            raise PermissionError("Set accept_cms_api_license: true in configs/settings.yaml after reading the license terms.")
        if self._token is None:
            data = self.http.get_json(f"{API}/metadata/license-agreement/", use_cache=False)
            self._token = data["data"][0]["Token"]
        return self._token

    def revision_history(self, lcd_id: int, ver: int) -> list[dict]:
        url = f"{API}/data/lcd/revision-history/?lcdid={lcd_id}&ver={ver}"
        res = self.http.get_json(url, headers={"Authorization": f"Bearer {self.token()}"})
        out = []
        for row in res.get("data", []):
            text = html.unescape(html.unescape(row.get("rev_hist_exp") or ""))
            row["rev_hist_text"] = re.sub(r"<[^>]+>", " ", text).strip()
            out.append(row)
        return out
