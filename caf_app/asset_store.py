# caf_app/asset_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def project_root() -> Path:
    # caf_app/asset_store.py -> caf_app -> project root
    return Path(__file__).resolve().parents[1]


def campaign_root(slug: str) -> Path:
    return project_root() / "campaigns" / slug


def images_root(slug: str) -> Path:
    return campaign_root(slug) / "images"


def assets_index_path(slug: str) -> Path:
    return campaign_root(slug) / "assets_index.json"


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_mtime_iso(path: Path) -> str:
    ts = path.stat().st_mtime
    return _utc_iso(datetime.fromtimestamp(ts, tz=timezone.utc))


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _infer_kind(relpath: str) -> str:
    s = relpath.lower().replace("\\", "/")
    if "/uploaded/" in s or "/external/" in s:
        return "external"
    if "/edited/" in s:
        return "edited"
    if "/variants/" in s or "/variant/" in s:
        return "variant"
    # generated can contain both, but defaulting to origin is fine for MVP
    return "origin"


@dataclass
class AssetRec:
    asset_id: str
    filename: str
    relpath: str
    kind: str
    created_at: str
    # optional lineage hints
    base_image: Optional[str] = None
    parent_id: Optional[str] = None
    root_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "relpath": self.relpath,
            "kind": self.kind,
            "created_at": self.created_at,
            "base_image": self.base_image,
            "parent_id": self.parent_id,
            "root_id": self.root_id,
        }


def load_assets_index(slug: str) -> Dict[str, Dict[str, Any]]:
    p = assets_index_path(slug)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_assets_index(slug: str, index: Dict[str, Dict[str, Any]]) -> None:
    p = assets_index_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def scan_images(slug: str) -> List[Path]:
    root = images_root(slug)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if _is_image(p)]


def _filename_to_existing_asset_id(existing_index: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Preserve IDs across reruns by mapping filename -> asset_id from the existing index.
    If you ever allow duplicate filenames in a campaign, you’ll want to key by relpath instead.
    """
    out: Dict[str, str] = {}
    for asset_id, rec in existing_index.items():
        fn = rec.get("filename")
        if fn and fn not in out:
            out[fn] = asset_id
    return out


def rebuild_assets_index(slug: str) -> List[str]:
    """
    Deterministically rebuild assets_index.json from disk + existing index hints.
    Returns warnings (strings).
    """
    warnings: List[str] = []
    camp = campaign_root(slug)
    if not camp.exists():
        raise FileNotFoundError(f"Campaign not found: {camp}")

    existing = load_assets_index(slug)
    filename_to_id = _filename_to_existing_asset_id(existing)

    files = scan_images(slug)
    recs: List[AssetRec] = []

    for f in files:
        rel = str(f.relative_to(camp)).replace("\\", "/")
        fn = f.name

        prior_id = filename_to_id.get(fn)
        prior = existing.get(prior_id, {}) if prior_id else {}

        asset_id = prior_id or str(uuid4())
        created_at = prior.get("created_at") or _file_mtime_iso(f)
        kind = prior.get("kind") or _infer_kind(rel)

        rec = AssetRec(
            asset_id=asset_id,
            filename=fn,
            relpath=rel,
            kind=kind,
            created_at=created_at,
            base_image=prior.get("base_image"),
            parent_id=prior.get("parent_id"),
            root_id=prior.get("root_id"),
        )
        recs.append(rec)

    # lineage repair (MVP)
    by_filename = {r.filename: r for r in recs}

    for r in recs:
        if r.kind == "origin":
            r.root_id = r.asset_id
            r.parent_id = None

    for r in recs:
        if r.kind != "variant":
            continue
        if not r.base_image:
            # Leave as-is; still valid to index variants without lineage
            warnings.append(f"Variant {r.filename} has no base_image; lineage not linked.")
            if not r.root_id:
                r.root_id = r.asset_id
            continue

        base = by_filename.get(r.base_image)
        if not base:
            warnings.append(f"Variant {r.filename} base_image {r.base_image} not found on disk; lineage not linked.")
            if not r.root_id:
                r.root_id = r.asset_id
            r.parent_id = None
            continue

        # ensure base root is set
        if not base.root_id:
            base.root_id = base.asset_id

        r.parent_id = base.asset_id
        r.root_id = base.root_id

    # write index keyed by asset_id
    out = {r.asset_id: r.to_dict() for r in sorted(recs, key=lambda x: (x.created_at, x.filename))}
    save_assets_index(slug, out)
    return warnings
