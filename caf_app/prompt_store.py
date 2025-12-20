# caf_app/prompt_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import re
import uuid

def attach_prompt_to_image_meta(
    meta: dict,
    filename: str,
    *,
    prompt_id: str,
    prompt_source: str,
    parent_prompt_id: str | None = None,
) -> None:
    """
    Mutates `meta` in-place: attaches prompt linkage fields to a single image record.
    Assumes meta[filename] is the per-image metadata dict.
    """
    if filename not in meta or not isinstance(meta[filename], dict):
        meta[filename] = {}

    meta[filename]["prompt_id"] = prompt_id
    meta[filename]["prompt_source"] = prompt_source
    meta[filename]["parent_prompt_id"] = parent_prompt_id

def normalize_prompt(text: str) -> str:
    """
    Normalize prompt text for deduplication.
    MVP rules:
    - strip leading/trailing whitespace
    - collapse internal whitespace
    """
    if not text:
        return ""

    text = text.strip()
    # collapse all whitespace to single spaces
    text = re.sub(r"\s+", " ", text)
    return text

def find_prompt_by_text(
    prompts_index: dict,
    prompt_text: str,
) -> dict | None:
    """
    Return existing prompt record whose normalized prompt_text matches.
    """
    target = normalize_prompt(prompt_text)
    if not target:
        return None

    for record in prompts_index.get("prompts", {}).values():
        if normalize_prompt(record.get("prompt_text", "")) == target:
            return record

    return None

def upsert_prompt_record(
    slug: str,
    *,
    prompt_text: str,
    input_text: str | None = None,
    source: str = "manual",   # manual | reuse | refine | imported
    parent_prompt_id: str | None = None,
) -> str:
    """
    Create or reuse a prompt record in the campaign-local prompt index.

    Returns:
        prompt_id (str)
    """
    data = load_prompts_index(slug)
    prompts = data["prompts"]

    normalized = normalize_prompt(prompt_text)
    if not normalized:
        raise ValueError("prompt_text cannot be empty")

    # ---- Deduplication ----
    existing = find_prompt_by_text(data, normalized)

    now = _utc_now_iso()

    if existing:
        # Reuse existing prompt
        existing["usage_count"] = int(existing.get("usage_count", 0)) + 1
        existing["last_used_at"] = now
        existing["updated_at"] = now

        save_prompts_index(slug, data)
        return existing["prompt_id"]

    # ---- Create new prompt record ----
    prompt_id = str(uuid.uuid4())

    record = {
        "prompt_id": prompt_id,
        "created_at": now,
        "updated_at": now,
        "prompt_text": normalized,
        "input_text": input_text,
        "source": source,
        "parent_prompt_id": parent_prompt_id,
        "usage_count": 1,
        "last_used_at": now,
        "favorite": False,
    }

    prompts[prompt_id] = record
    save_prompts_index(slug, data)

    return prompt_id


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
