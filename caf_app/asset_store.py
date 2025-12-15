from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def load_assets_index(slug: str, campaigns_root: Path = Path("campaigns")) -> dict[str, dict]:
    p = campaigns_root / slug / "assets_index.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_assets_index(slug: str, index: dict[str, dict], campaigns_root: Path = Path("campaigns")) -> None:
    p = campaigns_root / slug / "assets_index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")

def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_mtime_iso(path: Path) -> str:
    ts = path.stat().st_mtime
    return _utc_iso(datetime.fromtimestamp(ts, tz=timezone.utc))


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def rebuild_assets_index(slug: str, campaigns_root: Path = Path("campaigns")) -> list[str]:
    """
    Minimal deterministic reindexer:
      - scans campaigns/<slug>/images/**
      - writes campaigns/<slug>/assets_index.json
      - preserves asset_id across reruns (relpath first, filename fallback)
    """
    warnings: list[str] = []
    camp_root = campaigns_root / slug
    if not camp_root.exists():
        raise FileNotFoundError(f"Campaign not found: {camp_root}")

    images_root = camp_root / "images"
    out_path = camp_root / "assets_index.json"

    if not images_root.exists():
        out_path.write_text(json.dumps({}, indent=2, sort_keys=True), encoding="utf-8")
        return warnings

    # ---- load existing index (if any) ------------------------------------

    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # ---- build ID preservation maps --------------------------------------

    relpath_to_id: dict[str, str]


# ---------------------------------------------------------------------------
# Compatibility shim: AssetStore (used by your UI)
# ---------------------------------------------------------------------------

class AssetStore:
    """
    Backward-compatible AssetStore wrapper.

    Existing usage:
        ASSET_STORE = AssetStore(campaigns_root=Path("campaigns"))

    Before using record operations, call:
        ASSET_STORE.set_campaign(slug)
    """

    def __init__(self, campaigns_root: Path, slug: str | None = None):
        self.campaigns_root = campaigns_root
        self.slug = slug
        self._index: dict[str, dict] | None = None

    def set_campaign(self, slug: str) -> None:
        if slug != self.slug:
            self.slug = slug
            self._index = None

    def _campaign_root(self) -> Path:
        if not self.slug:
            raise RuntimeError("AssetStore.slug is not set")
        return self.campaigns_root / self.slug

    def _assets_index_path(self) -> Path:
        return self._campaign_root() / "assets_index.json"

    def load(self) -> dict[str, dict]:
        if self._index is None:
            self._index = load_assets_index(self.slug, campaigns_root=self.campaigns_root) if self.slug else {}
        return self._index


        p = self._assets_index_path()
        if not p.exists():
            self._index = {}
        else:
            self._index = json.loads(p.read_text(encoding="utf-8"))
        return self._index

    def save(self) -> None:
        if not self.slug or self._index is None:
            return
        save_assets_index(self.slug, self._index, campaigns_root=self.campaigns_root)

    def find_by_filename(self, filename: str) -> dict | None:
        for rec in self.load().values():
            if rec.get("filename") == filename:
                return rec
        return None

    def upsert(self, record: dict) -> str:
        if not self.slug:
            raise RuntimeError("AssetStore.slug is not set")
        if not isinstance(record, dict):
            raise TypeError("record must be a dict")

        filename = record.get("filename")
        relpath = record.get("relpath")
        if not filename or not relpath:
            raise ValueError("record must include filename and relpath")

        idx = self.load()
        existing = self.find_by_filename(filename)
        asset_id = record.get("asset_id") or (existing.get("asset_id") if existing else None) or str(uuid4())
        record["asset_id"] = asset_id

        record.setdefault("kind", "origin")
        record.setdefault("created_at", _file_mtime_iso(self._campaign_root() / relpath))

        if record.get("kind") == "origin":
            record.setdefault("root_id", asset_id)
            record.setdefault("parent_id", None)

        if asset_id in idx:
            idx[asset_id] = {**idx[asset_id], **record}
        else:
            idx[asset_id] = record

        return asset_id

    def rebuild_index(self) -> list[str]:
        if not self.slug:
            raise RuntimeError("AssetStore.slug is not set")
        return rebuild_assets_index(self.slug, campaigns_root=self.campaigns_root)
