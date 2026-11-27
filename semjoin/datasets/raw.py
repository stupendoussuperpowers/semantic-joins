"""Cache for downloaded source files, under ``data/raw/<source>/``.

Downloads (``semjoin download-<source>``) are the only network access. They
write each file once into the cache and records where it came from in
``data/raw/<source>/fetch.json``:

    {"files": {"<relative path>": {"url": ..., "sha256": ..., "bytes": ...,
                                   "fetched_at": ...}},
     "pinned": {...}}          # e.g. a commit hash or query date for the source

Builders then read only from the cache, so rebuilding
tasks, running tests and validating never need the network. ``verify``
re-hashes cached files so a corrupted or hand-edited cache is caught.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST = "fetch.json"
# A generic client name; never put personal details here.
USER_AGENT = "semjoin-dataset-fetcher/0.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class RawCache:
    def __init__(self, data_root: Path, source: str):
        self.source = source
        self.root = data_root / "raw" / source
        self._manifest_path = self.root / MANIFEST

    def manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {"files": {}, "pinned": {}}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest["files"] = dict(sorted(manifest["files"].items()))
        self._manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

    def pin(self, **values: Any) -> None:
        """Record what version of the source the cache holds (commit, query date, ...)."""
        manifest = self.manifest()
        manifest["pinned"].update(values)
        self._write_manifest(manifest)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def has(self, rel: str) -> bool:
        return rel in self.manifest()["files"] and self.path(rel).exists()

    def put(self, rel: str, data: bytes, url: str | None = None) -> Path:
        """Store bytes in the cache and record their origin and hash."""
        out = self.path(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        manifest = self.manifest()
        manifest["files"][rel] = {
            "url": url,
            "sha256": _sha256(data),
            "bytes": len(data),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_manifest(manifest)
        return out

    def fetch(
        self,
        rel: str,
        url: str,
        *,
        refresh: bool = False,
        headers: dict[str, str] | None = None,
        getter=_http_get,
    ) -> Path:
        """Download ``url`` to ``rel`` unless it is already cached."""
        if self.has(rel) and not refresh:
            return self.path(rel)
        print(f"fetch {self.source}/{rel}", file=sys.stderr)
        return self.put(rel, getter(url, headers), url=url)

    def require(self, rel: str) -> Path:
        """Path to a cached file, or a clear error telling the user to fetch it."""
        if not self.has(rel):
            raise FileNotFoundError(
                f"{self.path(rel)} is not cached; run `semjoin download-{self.source}` first"
            )
        return self.path(rel)

    def verify(self) -> list[str]:
        """Return a problem description per cached file that is missing or altered."""
        problems = []
        for rel, entry in self.manifest()["files"].items():
            p = self.path(rel)
            if not p.exists():
                problems.append(f"{self.source}/{rel}: missing")
            elif _sha256(p.read_bytes()) != entry["sha256"]:
                problems.append(f"{self.source}/{rel}: sha256 mismatch")
        return problems
