# caf_app/prompt_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---- Adjust this if your project uses a different campaigns root ----
# If you already have a helper like campaign_dir(slug) in caf_app.storage, use that instead.
def campaigns_root() -> Path:
    return Path("campaigns")


def campaign_dir(slug: str) -> Path:
    return campaigns_root() / slug


def _utc_now_iso() -> str:
    # e.g. "2025-12-20T20:10:12Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def campaign_prompts_path(slug: str) -> Path:
    return campaign_dir(slug) / "prompts_index.json"


def _default_prompts_index() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _utc_now_iso(),
        "prompts": {},  # prompt_id -> PromptRecord dict
    }


def load_prompts_index(slug: str) -> Dict[str, Any]:
    """
    Load the campaign-local prompt library index.
    Returns a dict with keys: schema_version, updated_at, prompts.
    Never raises for 'file not found' (returns empty default instead).
    """
    path = campaign_prompts_path(slug)
    if not path.exists():
        return _default_prompts_index()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Corrupt JSON: fail soft for MVP (you can add logging later)
        return _default_prompts_index()

    # Minimal validation / normalization (MVP)
    if not isinstance(data, dict):
        return _default_prompts_index()

    schema_version = data.get("schema_version")
    prompts = data.get("prompts")

    if schema_version != 1 or not isinstance(prompts, dict):
        # Unknown schema or wrong shape: start fresh (MVP)
        return _default_prompts_index()

    # Ensure updated_at exists (optional)
    if not isinstance(data.get("updated_at"), str):
        data["updated_at"] = _utc_now_iso()

    return data


def save_prompts_index(slug: str, data: Dict[str, Any]) -> None:
    """
    Save the prompt library index atomically-ish (write_text is fine for MVP).
    Ensures campaign dir exists. Updates updated_at.
    """
    if not isinstance(data, dict):
        raise TypeError("prompts_index data must be a dict")

    # enforce minimal shape
    if data.get("schema_version") != 1:
        data["schema_version"] = 1
    if "prompts" not in data or not isinstance(data["prompts"], dict):
        data["prompts"] = {}

    data["updated_at"] = _utc_now_iso()

    path = campaign_prompts_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
