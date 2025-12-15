# caf_app/asset_store.py
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Utilities
# ----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, data: Any) -> None:
    """
    Atomic JSON writer: writes to a temp file then os.replace().
    Prevents partial/corrupt writes if the process dies mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v in (0, 1):
        return bool(v)
    return default


# ----------------------------
# Data contracts (lightweight)
# ----------------------------

@dataclass(frozen=True)
class AssetSummary:
    asset_id: str
    family_id: str
    kind: str                      # origin | variant | resize | edit | external
    created_at: str
    is_current: bool
    favorite: bool
    image_path: str                # relative path from campaign root (recommended)
    thumb_path: Optional[str] = None
    parent_id: Optional[str] = None


class AssetStore:
    """
    File-based asset metadata store (multi-file JSON):
      - assets/index.json  : list[AssetSummary-like dict]
      - assets/items/*.json: full asset records

    This store is UI-agnostic. Your existing UI can keep using the old meta.json
    until you switch it to read summaries from this store.
    """

    def __init__(self, campaigns_root: Path) -> None:
        self.campaigns_root = campaigns_root

    # -------- Path helpers --------

    def campaign_dir(self, slug: str) -> Path:
        return self.campaigns_root / slug

    def assets_dir(self, slug: str) -> Path:
        return self.campaign_dir(slug) / "assets"

    def items_dir(self, slug: str) -> Path:
        return self.assets_dir(slug) / "items"

    def index_path(self, slug: str) -> Path:
        return self.assets_dir(slug) / "index.json"

    def asset_path(self, slug: str, asset_id: str) -> Path:
        return self.items_dir(slug) / f"{asset_id}.json"

    # -------- CRUD --------

    def list_summaries(self, slug: str) -> List[AssetSummary]:
        raw = _read_json(self.index_path(slug), default=[])
        out: List[AssetSummary] = []
        for r in raw:
            try:
                out.append(
                    AssetSummary(
                        asset_id=str(r["asset_id"]),
                        family_id=str(r["family_id"]),
                        kind=str(r.get("kind", "origin")),
                        created_at=str(r.get("created_at", "")),
                        is_current=_as_bool(r.get("is_current", True), True),
                        favorite=_as_bool(r.get("favorite", False), False),
                        image_path=str(r.get("image_path", "")),
                        thumb_path=r.get("thumb_path"),
                        parent_id=r.get("parent_id"),
                    )
                )
            except Exception:
                # Skip malformed entries rather than crashing the app.
                continue
        return out

    def get_asset(self, slug: str, asset_id: str) -> Optional[Dict[str, Any]]:
        p = self.asset_path(slug, asset_id)
        if not p.exists():
            return None
        return _read_json(p, default=None)

    def upsert_asset(self, slug: str, asset: Dict[str, Any]) -> str:
        """
        Writes full asset to assets/items/<asset_id>.json
        Updates/creates summary row in assets/index.json
        Returns asset_id.
        """
        asset_id = str(asset.get("asset_id") or uuid.uuid4())
        asset["asset_id"] = asset_id

        # Ensure family_id defaults: origin assets get family_id == asset_id
        kind = str(asset.get("kind", "origin"))
        if not asset.get("family_id"):
            asset["family_id"] = asset_id if kind == "origin" else asset.get("parent_family_id")  # fallback
        if not asset.get("family_id"):
            # Last resort: treat as its own family
            asset["family_id"] = asset_id

        asset.setdefault("created_at", _utc_now_iso())
        asset.setdefault("is_current", True)
        asset.setdefault("favorite", False)

        # Write full asset record
        _atomic_write_json(self.asset_path(slug, asset_id), asset)

        # Update index
        self._upsert_index_row(slug, asset)
        return asset_id

    def delete_asset(self, slug: str, asset_id: str) -> None:
        # remove item file
        p = self.asset_path(slug, asset_id)
        if p.exists():
            p.unlink()

        # remove from index
        idx = _read_json(self.index_path(slug), default=[])
        idx2 = [r for r in idx if str(r.get("asset_id")) != asset_id]
        _atomic_write_json(self.index_path(slug), idx2)

    # -------- Index management --------

    def rebuild_index(self, slug: str) -> Tuple[int, int]:
        """
        Rebuilds assets/index.json by scanning assets/items/*.json.
        Returns (num_assets, num_skipped).
        """
        items = self.items_dir(slug)
        items.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        skipped = 0

        for p in sorted(items.glob("*.json")):
            try:
                asset = _read_json(p, default=None)
                if not isinstance(asset, dict):
                    skipped += 1
                    continue
                rows.append(self._summary_from_asset(asset))
            except Exception:
                skipped += 1

        # Sort newest first (use created_at lexicographically if ISO)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        _atomic_write_json(self.index_path(slug), rows)
        return (len(rows), skipped)

    # -------- Internal helpers --------

    def _upsert_index_row(self, slug: str, asset: Dict[str, Any]) -> None:
        idx_path = self.index_path(slug)
        idx = _read_json(idx_path, default=[])
        asset_id = str(asset["asset_id"])
        new_row = self._summary_from_asset(asset)

        replaced = False
        for i, r in enumerate(idx):
            if str(r.get("asset_id")) == asset_id:
                idx[i] = new_row
                replaced = True
                break
        if not replaced:
            idx.append(new_row)

        # Keep index sorted by created_at desc
        idx.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        _atomic_write_json(idx_path, idx)

    def _summary_from_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "asset_id": str(asset.get("asset_id", "")),
            "family_id": str(asset.get("family_id", "")),
            "parent_id": asset.get("parent_id"),
            "kind": asset.get("kind", "origin"),
            "created_at": asset.get("created_at", ""),
            "is_current": bool(asset.get("is_current", True)),
            "favorite": bool(asset.get("favorite", False)),
            "image_path": asset.get("image_path", ""),
            "thumb_path": asset.get("thumb_path"),
        }
