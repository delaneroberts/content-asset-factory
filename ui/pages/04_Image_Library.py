# ui/pages/04_Image_Library.py
from __future__ import annotations
from caf_app.services.prompt_refiner import refine_prompt
import base64
import os
import time
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4
import io
import json
import zipfile
import requests
import streamlit as st
import uuid
from openai import OpenAI, OpenAIError
from huggingface_hub import InferenceClient
from PIL import Image
from caf_app.storage import load_campaign
from caf_app.models import Campaign  # for type hints / future use
from caf_app.asset_store import AssetStore
from caf_app.asset_store import load_assets_index
from textwrap import dedent
from typing import Dict, List, Any, List, Tuple, Optional
from openai import OpenAI
from typing import Dict, List
from pathlib import Path
from caf_app.prompt_store import upsert_prompt_record





ASSET_STORE = AssetStore(campaigns_root=Path("campaigns"))

# ---- Custom CSS for gallery improvements ----

#############
st.markdown("""
<style>
/* Add horizontal and vertical spacing between image cards */
.image-card {
    padding: 6px 12px 16px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    border-radius: 10px;
    border: 1px solid #e5e7eb;  /* default subtle border */
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}

/* Gold border for favorites */
.image-card.favorite {
    border: 3px solid #fbbf24;      /* gold-ish */
    box-shadow: 0 0 0 2px rgba(251, 191, 36, 0.35);
}

/* Container around the image */
.image-wrapper {
    position: relative;
    display: inline-block;
}

/* (Removed .favorite-badge – we don't need the floating star anymore) */
</style>
""", unsafe_allow_html=True)

##########
# ---------------------------------------------------------------------------
# Config / clients
# ---------------------------------------------------------------------------

client = OpenAI()

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_NANOBANANA_MODEL_ID = os.getenv("HF_NANOBANANA_MODEL_ID")


# ---------------------------------------------------------------------------
# Superceed, versioning helpers
# ---------------------------------------------------------------------------

#def family_key_for(p: Path) -> str:
#    info = merged_info(p.name)
#    # Prefer root_id from assets_index; fallback to asset_id; then filename
#    return str(info.get("root_id") or info.get("asset_id") or p.name)

def _auto_promote_current_after_delete(meta: dict, affected_root_ids: set[str]) -> dict:
    """
    If the current asset in a lineage was deleted, promote the highest-version
    remaining sibling (same root_id) to is_current=True.
    """
    for rid in affected_root_ids:
        # gather remaining siblings
        siblings = []
        for fn, info in meta.items():
            if not isinstance(info, dict):
                continue
            root = info.get("root_id") or info.get("asset_id")
            if root == rid:
                siblings.append((fn, info))

        if not siblings:
            continue  # lineage fully deleted

        # pick highest version (default 1 if missing)
        def vnum(item):
            info = item[1]
            try:
                return int(info.get("version", 1))
            except Exception:
                return 1

        best_fn, best_info = max(siblings, key=vnum)

        # set everyone else not-current, best current
        for fn, info in siblings:
            info["is_current"] = (fn == best_fn)
            meta[fn] = info

    return meta



def _supersede_asset(meta: dict, old_asset_id: str, new_filename: str) -> dict:
    """
    Mark a new file as the next version of an existing asset.
    """
    # Find the old record by asset_id
    old_name = None
    old_info = None
    for fn, info in meta.items():
        if isinstance(info, dict) and info.get("asset_id") == old_asset_id:
            old_name, old_info = fn, info
            break

    if not old_info:
        return meta  # nothing to do

    # Flip old to not-current
    old_info["is_current"] = False
    meta[old_name] = old_info

    # Initialize new (assumes it already exists in meta)
    new_info = meta.get(new_filename, {})
    if not isinstance(new_info, dict):
        new_info = {}

    new_info["root_id"] = old_info.get("root_id") or old_info.get("asset_id")
    new_info["supersedes_id"] = old_info.get("asset_id")
    new_info["version"] = int(old_info.get("version", 1)) + 1
    new_info["is_current"] = True

    meta[new_filename] = new_info
    return meta

def _apply_supersede(meta: dict, new_filename: str) -> dict:
    """
    Given metadata where new_filename already exists (and has root_id/asset_id),
    make it the new current version for its root_id lineage.
    """
    new_info = meta.get(new_filename, {})
    if not isinstance(new_info, dict):
        return meta

    root_id = new_info.get("root_id") or new_info.get("asset_id")
    if not root_id:
        return meta

    # --- HARDENING: find ALL currents in this lineage ---
    current_candidates = []
    for fn, info in meta.items():
        if not isinstance(info, dict):
            continue
        rid = info.get("root_id") or info.get("asset_id")
        if rid == root_id and info.get("is_current") is True:
            current_candidates.append((fn, info))

    # Choose the highest-version current if multiple exist
    current_fn = None
    current_info = None
    if current_candidates:
        current_fn, current_info = max(
            current_candidates,
            key=lambda x: int(x[1].get("version", 1)),
        )

    # If new is already current, nothing to do (before clearing flags)
    if current_fn == new_filename:
        return meta

    # Clear ALL current flags in this lineage (defensive)
    for fn, info in meta.items():
        if not isinstance(info, dict):
            continue
        rid = info.get("root_id") or info.get("asset_id")
        if rid == root_id and info.get("is_current") is True:
            info["is_current"] = False
            meta[fn] = info

    # If none marked current, don’t guess too much—just keep new as v1 current
    if not current_info:
        new_info.setdefault("version", 1)
        new_info["is_current"] = True
        meta[new_filename] = new_info
        return meta

    # Promote new
    new_info["supersedes_id"] = current_info.get("asset_id")
    new_info["version"] = int(current_info.get("version", 1)) + 1
    new_info["is_current"] = True
    meta[new_filename] = new_info

    return meta

# ---------------------------------------------------------------------------
# multiref + image generation helpers
# ---------------------------------------------------------------------------

def _resolve_campaign_image_path(slug: str, filename: str) -> Optional[Path]:
    img_dir = Path("campaigns") / slug / "images"
    # add/remove folders to match your CAF
    for sub in ["generated", "uploaded", "external", "variants", "references"]:
        p = img_dir / sub / filename
        if p.exists():
            return p
    p = img_dir / filename
    return p if p.exists() else None


def _campaign_dir(slug: str) -> Path:
    return Path("campaigns") / slug

def _images_dir(slug: str) -> Path:
    return _campaign_dir(slug) / "images"

def _refs_dir(slug: str) -> Path:
    # Keep multiref uploads separate so you can distinguish them from “real” library images later.
    return _images_dir(slug) / "references"

def _generated_dir(slug: str) -> Path:
    return _images_dir(slug) / "generated"

def _meta_path(slug: str) -> Path:
    # Adjust if you already have a different metadata filename
    return _images_dir(slug) / "images_meta.json"

def _load_meta(slug: str) -> Dict[str, Any]:
    p = _meta_path(slug)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_meta(slug: str, meta: Dict[str, Any]) -> None:
    p = _meta_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

def _safe_ext(filename: str) -> str:
    ext = (Path(filename).suffix or "").lower()
    return ext if ext in [".png", ".jpg", ".jpeg", ".webp"] else ".png"

def _save_uploaded_file(slug: str, uploaded, subdir: Path) -> str:
    subdir.mkdir(parents=True, exist_ok=True)
    out_name = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}{_safe_ext(uploaded.name)}"
    out_path = subdir / out_name
    out_path.write_bytes(uploaded.getbuffer())
    return out_name

def _build_multiref_prompt(user_prompt: str, roles: Dict[str, str], primary_name: str) -> str:
    # Keep it simple + explicit (MVP)
    role_lines = []
    for fname, role in roles.items():
        if fname == primary_name:
            continue
        if role and role != "(none)":
            role_lines.append(f"- Use {fname} as a {role} reference (do not copy identity).")

    role_block = "\n".join(role_lines) if role_lines else "- Other images are general style references only."

    return (
        "You are generating a NEW image.\n"
        f"Primary identity image: {primary_name} (preserve facial identity, proportions, age).\n"
        "Do NOT change gender/age/ethnicity.\n"
        "Supporting references:\n"
        f"{role_block}\n\n"
        "User request:\n"
        f"{user_prompt.strip()}\n\n"
        "Output requirements:\n"
        "- Photorealistic\n"
        "- Clean, professional look\n"
        "- No text, logos, watermarks\n"
    )


# ---------------------------------------------------------------------------
# Path + metadata helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """
    Return the project root directory (one level above /ui).
    This file lives at <root>/ui/pages/04_Image_Library.py
    """
    return Path(__file__).resolve().parents[2]


def _campaigns_root() -> Path:
    root = _project_root()
    campaigns = root / "campaigns"
    campaigns.mkdir(parents=True, exist_ok=True)
    return campaigns


def _metadata_path(slug: str) -> Path:
    """
    JSON file that tracks pinned / favorite status and other metadata
    per image for this campaign.
    """
    base = _campaigns_root() / slug
    base.mkdir(parents=True, exist_ok=True)
    return base / "image_metadata.json"


def _load_image_metadata(slug: str) -> dict:
    """
    Load image metadata for a campaign, keyed by filename.
    Example:
        {
          "123456.png": {
              "pinned": true,
              "favorite": false,
              "engine": "openai",
              "prompt": "...",
              "kind": "generated",
              "created_at": "2025-12-10T22:15:02Z"
          },
          ...
        }
    """
    path = _metadata_path(slug)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        # If anything goes wrong, fall back to empty metadata rather than dying.
        return {}


def _save_image_metadata(slug: str, metadata: dict) -> None:
    """
    Persist image metadata to disk.
    """
    path = _metadata_path(slug)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

# INVARIANT:
# - asset_id never changes
# - root_id identifies the lineage origin
# - parent_id points to the immediate ancestor (if variant)
# - exactly ONE asset per root_id has is_current == True
def _update_image_metadata_entry(
    slug: str,
    filename: str,
    *,
    kind: str,
    created_at: str | None = None,
    asset_id: str | None = None,
    family_id: str | None = None,
    parent_id: str | None = None,
    engine: str | None = None,
    prompt: str | None = None,
    **extra,
) -> None:
    meta = _load_image_metadata(slug)
    info = meta.get(filename, {})
    if not isinstance(info, dict):
        info = {}

    origin_like = {"origin", "uploaded", "external"}
    derivative_like = {"variant", "edit", "resize"}

    # Always set / normalize core fields
    info["kind"] = kind
    if created_at:
        info["created_at"] = created_at

    if asset_id:
        info["asset_id"] = asset_id

    # Persist provided family_id
    if family_id:
        info["family_id"] = family_id

    # Defaults for origin-like
    if kind in origin_like and asset_id:
        info.setdefault("family_id", asset_id)
        info.setdefault("root_id", asset_id)
        info.setdefault("parent_id", None)

    # Defaults for derivative-like: root_id must follow the family, not the child’s asset_id
    if kind in derivative_like:
        fam = info.get("family_id") or family_id
        if fam:
            info["root_id"] = str(fam)

    # Parent linkage when provided (variants should always pass parent_id)
    if parent_id is not None:
        info["parent_id"] = parent_id

    # Optional common fields
    if engine is not None:
        info["engine"] = engine
    if prompt is not None:
        info["prompt"] = prompt

    # Extras (instructions, base_image, version, is_current, etc.)
    for k, v in extra.items():
        if v is not None:
            info[k] = v

    meta[filename] = info
    _save_image_metadata(slug, meta)



def _ensure_image_metadata_schema(meta: dict) -> tuple[dict, bool]:
    """
    Ensure each meta[filename] entry has required keys for lineage/versioning.
    Returns: (possibly-updated meta, changed_flag)
    """
    changed = False

    for filename, info in list(meta.items()):
        if not isinstance(info, dict):
            meta[filename] = {}
            info = meta[filename]
            changed = True

        # Stable identity
        if not info.get("asset_id"):
            info["asset_id"] = str(uuid4())
            changed = True

        # Root identity (origin lineage)
        if not info.get("root_id"):
            info["root_id"] = info["asset_id"]
            changed = True

        # Lineage defaults
        if "parent_id" not in info:
            info["parent_id"] = None
            changed = True

        if "derivation" not in info or not isinstance(info.get("derivation"), dict):
            info["derivation"] = {"type": "origin"} if info.get("parent_id") is None else {"type": "unknown"}
            changed = True

        # Versioning defaults
        if "version" not in info:
            info["version"] = 1
            changed = True

        if "is_current" not in info:
            info["is_current"] = True
            changed = True

        if "supersedes_id" not in info:
            info["supersedes_id"] = None
            changed = True

        # Status (aligns with your inspector)
        # Prefer existing approval_status if you already use it
        if "status" not in info:
            if info.get("approval_status") in ["draft", "approved", "rejected"]:
                info["status"] = info["approval_status"]
            else:
                info["status"] = "draft"
            changed = True

        # Spec placeholder (future)
        if "spec" not in info:
            info["spec"] = None
            changed = True

        meta[filename] = info

    return meta, changed

def _get_current_slug() -> str | None:
    """
    The Dashboard / other pages should set:
        st.session_state["current_campaign_slug"]
    """
    return st.session_state.get("current_campaign_slug")


def _campaign_dirs(slug: str) -> Tuple[Path, Path]:
    """
    Returns (images_dir, generated_dir) for a campaign.
    """
    base = _campaigns_root() / slug
    images_dir = base / "images"
    generated_dir = images_dir / "generated"
    images_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, generated_dir


def _list_all_images(slug: str) -> List[Path]:
    """
    Return all images (uploaded + generated) for a campaign as Paths,
    sorted newest-first by modification time.
    """
    images_dir, generated_dir = _campaign_dirs(slug)

    paths: List[Path] = []
    for directory in (images_dir, generated_dir):
        if directory.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                paths.extend(directory.glob(ext))

    # Deduplicate and sort
    unique = list({p.resolve(): p for p in paths}.values())
    unique.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _stability_variants_from_base(
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    Use Stability's edit/inpaint endpoint to generate variants starting from
    a base image. This tends to preserve more of the original structure than
    pure text-to-image.
    """
    if not STABILITY_API_KEY:
        raise RuntimeError("STABILITY_API_KEY is not set in the environment.")

    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",
    }

    # Full-white mask: allow edits anywhere, but diffusion still starts from the base.
    base = Image.open(base_image_path).convert("RGBA")
    mask = Image.new("L", base.size, color=255)
    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_bytes = mask_buf.getvalue()

    images: List[bytes] = []

    for _ in range(n_images):
        with open(base_image_path, "rb") as f:
            files = {
                "image": ("image.png", f, "image/png"),
                "mask": ("mask.png", mask_bytes, "image/png"),
            }
            data = {
                "prompt": instructions,
                "output_format": "png",
            }

            resp = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Stability edit failed: {resp.status_code} {resp.text[:500]}"
            )

        images.append(resp.content)

    return images


def _openai_variants_from_base(
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    Use OpenAI's image edit API (gpt-image-1) to generate variants that
    preserve the original image as much as possible while applying the
    requested improvements.
    """
    images: List[bytes] = []

    for _ in range(n_images):
        with open(base_image_path, "rb") as f:
            try:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=f,
                    prompt=instructions,
                    n=1,
                    size="1024x1024",
                )
            except OpenAIError as e:
                raise RuntimeError(f"OpenAI variant edit failed: {e}") from e

        # gpt-image-1 always returns base64
        b64 = result.data[0].b64_json
        images.append(base64.b64decode(b64))

    return images


def _generate_openai_images(prompt: str, n_images: int) -> List[bytes]:
    if n_images < 1:
        return []

    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=n_images,
            size="1024x1024",
        )
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI image generation failed: {e}") from e

    images: List[bytes] = []
    for item in response.data:
        b64 = item.b64_json
        images.append(base64.b64decode(b64))
    return images


def _generate_stability_images(prompt: str, n_images: int) -> List[bytes]:
    if not STABILITY_API_KEY:
        raise RuntimeError("STABILITY_API_KEY is not set in the environment.")

    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/*",
    }

    images: List[bytes] = []
    for _ in range(n_images):
        files = {
            "prompt": (None, prompt),
            "output_format": (None, "png"),
        }
        resp = requests.post(url, headers=headers, files=files, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Stability API error {resp.status_code}: {resp.text[:500]}"
            )
        images.append(resp.content)
    return images


def _generate_nanobanana_images(prompt: str, n_images: int) -> List[bytes]:
    """
    NanoBanana / FLUX via Hugging Face InferenceClient.

    Uses HF_API_KEY as the token and HF_NANOBANANA_MODEL_ID as the model id,
    e.g. "black-forest-labs/FLUX.1-dev".
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set in the environment.")

    if not HF_NANOBANANA_MODEL_ID or "REPLACE_WITH" in HF_NANOBANANA_MODEL_ID:
        raise RuntimeError(
            "HF_NANOBANANA_MODEL_ID is not configured. "
            "Set it to your actual model id on Hugging Face, e.g. 'black-forest-labs/FLUX.1-dev'."
        )

    hf_client = InferenceClient(api_key=HF_API_KEY)

    images: List[bytes] = []
    for _ in range(n_images):
        img = hf_client.text_to_image(
            prompt=prompt,
            model=HF_NANOBANANA_MODEL_ID,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    return images


def _generate_images_for_engine(
    engine: str, prompt: str, n_images: int
) -> List[bytes]:
    engine = engine.lower()
    if engine == "openai":
        return _generate_openai_images(prompt, n_images)
    elif engine == "stability":
        return _generate_stability_images(prompt, n_images)
    elif engine == "nanobanana":
        return _generate_nanobanana_images(prompt, n_images)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def _generate_variants_for_engine(
    engine: str,
    base_image_path: Path,
    instructions: str,
    n_images: int,
) -> List[bytes]:
    """
    For now, only OpenAI is used for true image edits.
    """
    engine = engine.lower()
    if engine != "openai":
        raise RuntimeError("Variant-from-base currently supports only OpenAI image edits.")
    return _openai_variants_from_base(base_image_path, instructions, n_images)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _save_image_bytes(
    slug: str,
    img_bytes: bytes,
    generated: bool = True,
    *,
    kind: str = "origin",                 # origin | variant | resize | edit | external | uploaded
    engine: str | None = None,
    prompt: str | None = None,
    parent_asset_id: str | None = None,   # for variant/resize/edit
    family_id: str | None = None,         # pass for variant/resize/edit
    **extra,                               # e.g. instructions="...", base_image="..."
) -> tuple[Path, str | None]:
    """
    Save image bytes into the campaign folder and return (Path, asset_id).

    If generated=True, save under images/generated; else under images/.

    Choice A:
      1) Save bytes to disk
      2) Upsert to AssetStore (mint/get asset_id)
      3) Write legacy metadata INCLUDING asset_id/family_id/parent_id (+ any extra fields)
    """
    images_dir, generated_dir = _campaign_dirs(slug)
    target_dir = generated_dir if generated else images_dir

    ts = int(time.time() * 1000)
    filename = f"{ts}.png"
    path = target_dir / filename

    # 1) Save bytes to disk
    with open(path, "wb") as f:
        f.write(img_bytes)

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 2) Shadow-write to AssetStore (get asset_id)
    asset_id: str | None = None
    try:
        campaign_root = Path("campaigns") / slug
        rel_image_path = str(path.relative_to(campaign_root))

        record = {
            "kind": kind,
            "image_path": rel_image_path,
            "engine": engine,
            "prompt": prompt,
            "parent_id": parent_asset_id,
            "created_at": created_at,
        }

        if family_id:
            record["family_id"] = family_id

        # Optional extra fields can go into AssetStore too (safe; ignored if you don’t use them yet)
        for k, v in extra.items():
            if v is not None:
                record[k] = v

        asset_id = ASSET_STORE.upsert_asset(slug, record)

    except Exception:
        # Never break the UI because shadow-write failed
        asset_id = None

    # 3) Decide family_id for legacy meta:
    #    - Origins/uploads/external start a new family (family_id == asset_id)
    #    - Derivatives inherit provided family_id
    origin_like_kinds = {"origin", "uploaded", "external"}
    legacy_family_id = family_id
    if not legacy_family_id and asset_id and kind in origin_like_kinds:
        legacy_family_id = asset_id

    _update_image_metadata_entry(
        slug,
        filename,
        kind=kind,
        created_at=created_at,
        asset_id=asset_id,
        family_id=legacy_family_id,
        parent_id=parent_asset_id,
        engine=engine,
        prompt=prompt,
        **extra,  # write instructions/base_image/etc into legacy meta in the same call
    )

    # 4) Return both
    return path, asset_id

def _save_uploaded_image(slug: str, uploaded_file) -> tuple[Path, str | None]:
    """
    Save an uploaded image into the base images dir (not generated),
    then write legacy metadata WITH asset_id (Choice A), and shadow-write AssetStore.
    """
    images_dir, _ = _campaign_dirs(slug)
    suffix = Path(uploaded_file.name).suffix or ".png"
    ts = int(time.time() * 1000)
    filename = f"uploaded_{ts}{suffix}"
    dest = images_dir / filename

    # 1) Save bytes to disk
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 2) Shadow-write to AssetStore (get asset_id)
    asset_id = None
    try:
        campaign_root = Path("campaigns") / slug
        rel_image_path = str(dest.relative_to(campaign_root))

        asset_id = ASSET_STORE.upsert_asset(
            slug,
            {
                "kind": "external",     # or "uploaded" if you prefer
                "image_path": rel_image_path,
                "engine": "upload",
                "prompt": None,
                "parent_id": None,
                "created_at": created_at,
                "original_filename": uploaded_file.name,
            },
        )
    except Exception:
        pass

    # 3) Now write legacy metadata INCLUDING asset_id/family_id (Choice A)
    _update_image_metadata_entry(
        slug,
        filename,
        kind="uploaded",
        original_filename=uploaded_file.name,
        created_at=created_at,
        asset_id=asset_id,
        family_id=asset_id,  # uploaded/origin starts a new family
        parent_id=None,
        engine="upload",
    )

    # 4) Return both so call sites can use it if desired
    return dest, asset_id


def _build_images_zip(slug: str) -> bytes | None:
    """
    Package all images for a campaign into a ZIP file and return the bytes.
    """
    images = _list_all_images(slug)
    if not images:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, arcname=img_path.name)

    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _render_tools_panel(slug: str) -> None:
    """
    Compact wrapper that puts all the 'tool' sections behind a single expander
    with tabs, so the top of the page stays small.
    """
    with st.expander(
        "Tools: upload / generate / variants / export",
        expanded=False,
    ):
        tabs = st.tabs(
            [
                "Upload external images",
                "Generate from prompt",
                "Variants from base",
                "Multi-reference",
                "Export images",
            ]
        )


        with tabs[0]:
            _render_upload_section(slug)

        with tabs[1]:
            _render_prompt_generation_ui(slug)

        with tabs[2]:
            _render_variant_generation_ui(slug)

        with tabs[3]:
            _render_multiref_generation_ui(slug)

        with tabs[4]:
            _render_export_section(slug)

def _render_multiref_generation_ui(slug: str) -> None:
    st.markdown("### Create from multiple references")
    st.caption(
        "Use either (A) images you selected in the gallery, or (B) upload 2–6 reference images here."
    )

    # ----------------------------
    # A) Use selected images (from gallery)
    # ----------------------------
    selected: List[str] = st.session_state.get("selected_images", []) or []

    colA, colB = st.columns([1, 2])
    with colA:
        use_selected = st.button(
            "Use selected images",
            key=f"multiref_use_selected_{slug}",
            disabled=len(selected) < 2,
            help="Select at least 2 images in the gallery first.",
        )
    with colB:
        if selected:
            st.caption(f"Gallery selected: {len(selected)} image(s)")

    if use_selected:
        st.session_state[f"multiref_from_gallery_{slug}"] = {
            "primary": selected[0],
            "supporting": selected[1:6],  # cap total refs to 6
        }
        st.success("Loaded selections from gallery.")

    gallery_pick = st.session_state.get(f"multiref_from_gallery_{slug}")

    if gallery_pick:
        picked = [gallery_pick["primary"]] + list(gallery_pick.get("supporting", []))

        st.markdown("#### Gallery references")
        primary_name = st.selectbox(
            "Primary identity (required)",
            options=picked,
            index=0,
            key=f"multiref_gallery_primary_{slug}",
        )

        role_options = ["(none)", "pose", "lighting", "wardrobe", "mood"]
        roles: Dict[str, str] = {primary_name: "primary"}

        st.markdown("#### Optional: tag supporting references")
        for n in picked:
            if n == primary_name:
                continue
            roles[n] = st.selectbox(
                f"Role for `{n}`",
                options=role_options,
                index=0,
                key=f"multiref_gallery_role_{slug}_{n}",
            )

        # Persist back into the gallery_pick state (so Generate uses your choices)
        st.session_state[f"multiref_from_gallery_{slug}"] = {
            "primary": primary_name,
            "supporting": [n for n in picked if n != primary_name],
            "roles": roles,
        }

        st.info(
            f"Using gallery selections — Primary: `{primary_name}` "
            f"| Supporting: {len([n for n in picked if n != primary_name])}"
        )


    # ----------------------------
    # B) Upload reference images
    # ----------------------------
    uploads = st.file_uploader(
        "Upload reference images (2–6) — optional if you’re using gallery selections",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"multiref_uploads_{slug}",
    )

    # Must have either gallery refs OR >=2 uploads
    if (not uploads or len(uploads) < 2) and not gallery_pick:
        st.info("Select at least 2 images in the gallery OR upload at least 2 images here.")
        return

    # Cap uploads (only if uploads provided)
    if uploads and len(uploads) > 6:
        st.warning("Please limit to 6 uploaded images for now (MVP).")
        uploads = uploads[:6]

    use_gallery = bool(gallery_pick)
    use_uploads = bool(uploads) and len(uploads) >= 2

    if use_gallery and use_uploads:
        st.warning(
            "Both gallery selections and uploads are present. "
            "MVP behavior: using gallery selections and ignoring uploads."
        )

    # ----------------------------
    # If gallery mode: stop here for now (prevents half-wired behavior)
    # ----------------------------
    if use_gallery:
        prompt = st.text_area(
            "What should we generate?",
            value="Professional headshot, neutral background, natural light, realistic.",
            height=90,
            key=f"multiref_prompt_gallery_{slug}",
        ).strip()

        do_generate = st.button(
            "Generate",
            key=f"multiref_generate_gallery_{slug}",
            disabled=not prompt,
        )

        if not do_generate:
            return

        try:
            _generated_dir(slug).mkdir(parents=True, exist_ok=True)

            primary_name = gallery_pick["primary"]
            supporting_names = gallery_pick.get("supporting", [])

            # Resolve primary image path
            primary_path = _resolve_campaign_image_path(slug, primary_name)
            if not primary_path:
                st.error(f"Could not find primary image on disk: {primary_name}")
                return

            # Resolve supporting image paths
            supporting_ok = []
            supporting_paths = []
            for name in supporting_names:
                p = _resolve_campaign_image_path(slug, name)
                if not p:
                    st.warning(f"Skipping missing supporting image: {name}")
                    continue
                supporting_ok.append(name)
                supporting_paths.append(p)


            if not supporting_paths:
                st.error("Need at least one supporting image in addition to the primary.")
                return

            # Minimal roles for gallery MVP
            roles = {primary_name: "primary"}
            for n in supporting_names:
                roles[n] = "(none)"

            full_prompt = _build_multiref_prompt(
                prompt,
                roles,
                primary_name=primary_name,
            )

            client = OpenAI()

            img_paths = [primary_path] + supporting_paths
            files = [p.open("rb") for p in img_paths]
            try:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=files,
                    prompt=full_prompt,
                    input_fidelity="high",
                    size="1024x1024",
                    output_format="png",
                )
            finally:
                for f in files:
                    try:
                        f.close()
                    except Exception:
                        pass

            img_bytes = base64.b64decode(result.data[0].b64_json)

            out_name = f"{int(time.time() * 1000)}_multiref.png"
            out_path = _generated_dir(slug) / out_name
            out_path.write_bytes(img_bytes)

            meta = _load_meta(slug)
            meta[out_name] = {
                "kind": "generated",
                "engine": "openai",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "prompt": prompt,
                "instructions": full_prompt,
                "base_image": primary_name,
                "reference_images": (
                    [{"file": primary_name, "role": "primary"}]
                    + [{"file": n, "role": "(none)"} for n in supporting_names]
                ),
                "derivation": {"type": "multi_reference_guided", "source": "library"},
                "spec": {"size": "1024x1024", "output_format": "png", "input_fidelity": "high"},
            }
            _save_meta(slug, meta)

            st.success(f"Created: {out_name}")
            st.image(str(out_path), use_container_width=True)

        except Exception as e:
            st.error(f"Multi-reference (gallery) generation failed: {e}")

        return



        # ----------------------------
        # Upload mode UI (names + primary + roles)
        # ----------------------------
        # At this point, uploads must exist and have len >= 2
        names = [u.name for u in uploads]

        primary = st.selectbox(
            "Primary identity (required)",
            options=names,
            index=0,
            key=f"multiref_primary_upload_{slug}",
            help="This image’s identity should be preserved.",
        )

        role_options = ["(none)", "pose", "lighting", "wardrobe", "mood"]
        st.markdown("#### Optional: tag supporting references")
        roles: Dict[str, str] = {}

        for n in names:
            if n == primary:
                roles[n] = "primary"
                continue
            roles[n] = st.selectbox(
                f"Role for `{n}`",
                options=role_options,
                index=0,
                key=f"multiref_role_upload_{slug}_{n}",
            )

        prompt = st.text_area(
            "What should we generate?",
            value="Professional headshot, neutral background, natural light, realistic.",
            height=90,
            key=f"multiref_prompt_upload_{slug}",
        ).strip()

        with st.expander("Review selection", expanded=False):
            st.write("**Primary:**", primary)
            st.write("**Roles:**", roles)
            st.write("**Prompt:**", prompt)

        do_generate = st.button(
            "Generate",
            key=f"multiref_generate_upload_{slug}",
            disabled=not prompt,
        )
        if not do_generate:
            return

    # ----------------------------
    # Upload-mode Generate handler (your existing working logic)
    # ----------------------------
    try:
        _refs_dir(slug).mkdir(parents=True, exist_ok=True)
        _generated_dir(slug).mkdir(parents=True, exist_ok=True)

        # Save uploaded refs into campaigns/<slug>/images/references/
        saved_map: Dict[str, str] = {}
        for u in uploads:
            saved_map[u.name] = _save_uploaded_file(slug, u, _refs_dir(slug))

        primary_saved = saved_map[primary]

        full_prompt = _build_multiref_prompt(prompt, roles, primary_name=primary)

        client = OpenAI()  # expects OPENAI_API_KEY in env

        # Primary first, then others
        img_paths: List[Path] = [(_refs_dir(slug) / primary_saved)]
        for orig_name in names:
            if orig_name == primary:
                continue
            img_paths.append(_refs_dir(slug) / saved_map[orig_name])

        files = [p.open("rb") for p in img_paths]
        try:
            result = client.images.edit(
                model="gpt-image-1",
                image=files,
                prompt=full_prompt,
                input_fidelity="high",
                size="1024x1024",
                output_format="png",
            )
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

        img_bytes = base64.b64decode(result.data[0].b64_json)

        out_name = f"{int(time.time() * 1000)}_multiref.png"
        out_path = _generated_dir(slug) / out_name
        out_path.write_bytes(img_bytes)

        meta = _load_meta(slug)
        meta[out_name] = {
            "kind": "generated",
            "engine": "openai",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": prompt,
            "instructions": full_prompt,
            "base_image": primary_saved,
            "reference_images": [{"file": saved_map[n], "role": roles.get(n, "(none)")} for n in names],
            "derivation": {"type": "multi_reference_guided"},
            "spec": {"size": "1024x1024", "output_format": "png", "input_fidelity": "high"},
        }
        _save_meta(slug, meta)

        st.success(f"Created: {out_name}")
        st.image(str(out_path), use_container_width=True)

    except Exception as e:
        st.error(f"Multi-reference generation failed: {e}")


def _render_campaign_header(slug: str) -> None:
    campaign = load_campaign(slug)

    if campaign is None:
        st.warning(
            f"Could not load campaign for slug '{slug}'. "
            "It may have been deleted or the JSON file is missing."
        )
        return

    name = None
    for attr in ("name", "title", "campaign_name"):
        if hasattr(campaign, attr):
            name = getattr(campaign, attr)
            break

    if not name and isinstance(getattr(campaign, "model_dump", None), callable):
        data = campaign.model_dump()
        for key in ("name", "title", "campaign_name"):
            if key in data and data[key]:
                name = data[key]
                break

    if not name:
        name = slug

    st.subheader(f"Campaign: {name}")

    brief = None
    for attr in ("campaign_brief", "brief", "description", "summary"):
        if hasattr(campaign, attr):
            brief = getattr(campaign, attr)
            break

    if brief is None and isinstance(getattr(campaign, "model_dump", None), callable):
        data = campaign.model_dump()
        for key in ("campaign_brief", "brief", "description", "summary"):
            if key in data and data[key]:
                brief = data[key]
                break



def _render_upload_section(slug: str) -> None:
    st.markdown("### ⬆️ Upload External Images")

    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help=(
            "These images will appear in the gallery below. "
            "You can pin any of them and then use pinned images as bases for variants."
        ),
    )

    if not uploaded_files:
        return

    # Small preview / summary
    st.caption(f"{len(uploaded_files)} file(s) selected.")

    if st.button("Save uploaded images", key=f"save_uploads_{slug}"):
        saved = 0
        failed: list[str] = []

        for f in uploaded_files:
            try:
                dest, asset_id = _save_uploaded_image(slug, f)
                # dest.name is available if you want to log/debug, but not needed here
                saved += 1
            except Exception as e:
                failed.append(f"{getattr(f, 'name', 'unknown')}: {e}")

        if saved:
            st.success(f"Saved {saved} image(s). They now appear in the gallery.")

        if failed:
            st.warning("Some uploads failed:")
            for msg in failed[:8]:
                st.write(f"• {msg}")
            if len(failed) > 8:
                st.write(f"• ...and {len(failed) - 8} more")

        st.rerun()
def _render_variant_generation_ui(slug: str) -> None:
    st.markdown("### 🪄 Generate Variants From a Base Image")

    with st.expander("Open variant generator", expanded=False):
        all_images = _list_all_images(slug)

        meta = _load_image_metadata(slug)
        meta, ensured = _ensure_image_metadata_schema(meta)
        if ensured:
            _save_image_metadata(slug, meta)

        selected_images = [p for p in all_images if meta.get(p.name, {}).get("selected")]

        base_image_path: Path | None = None

        if selected_images:
            cols = st.columns(min(len(selected_images), 4))
            for i, p in enumerate(selected_images):
                with cols[i % len(cols)]:
                    st.image(str(p), caption=p.name, use_container_width=True)

            base_name = st.radio(
                "Base image",
                options=[p.name for p in selected_images],
                index=0,
            )
            base_image_path = next(p for p in selected_images if p.name == base_name)
        else:
            st.info("Select an image in the gallery to use as a base.")
            return

        instructions = st.text_area(
            "How should we improve this image?",
            height=100,
        )

        n_variants = st.slider("Number of variants", 1, 6, 2)

        supersede_current = st.checkbox(
            "Make this the new current version",
            value=False,
        )

        if not st.button("Generate improved variants", type="primary"):
            return

        if not instructions.strip():
            st.warning("Please enter instructions.")
            return

        base_info = meta.get(base_image_path.name, {})
        base_asset_id = base_info.get("asset_id")
        base_root_id = base_info.get("root_id") or base_asset_id

        if not base_asset_id or not base_root_id:
            st.error("Base image missing asset_id/root_id.")
            return

        # Determine next versions (Option A)
        def _next_versions(meta: dict, root_id: str, count: int) -> list[int]:
            max_v = 0
            for inf in meta.values():
                if not isinstance(inf, dict):
                    continue
                rid = inf.get("root_id") or inf.get("asset_id")
                if str(rid) == str(root_id):
                    try:
                        max_v = max(max_v, int(inf.get("version") or 0))
                    except Exception:
                        pass
            return list(range(max_v + 1, max_v + 1 + count))

        versions = _next_versions(meta, base_root_id, n_variants)

        with st.spinner("Generating variants…"):
            imgs = _generate_variants_for_engine(
                "openai",
                base_image_path,
                instructions.strip(),
                n_variants,
            )

        last_name = None

        for img_bytes, vnum in zip(imgs, versions):
            path, asset_id = _save_image_bytes(
                slug,
                img_bytes,
                generated=True,
                kind="variant",
                engine="openai",
                prompt=instructions.strip(),
                parent_asset_id=base_asset_id,
                family_id=base_root_id,
                base_image=base_image_path.name,
                version=vnum,
                is_current=False,
            )
            last_name = path.name

        if supersede_current and last_name:
            meta2 = _load_image_metadata(slug)
            meta2 = _apply_supersede(meta2, last_name)
            _save_image_metadata(slug, meta2)

        st.success("Variants created.")
        st.rerun()

# Only place prompt-based generation is done

def _attach_prompt_to_image_metadata(
    slug: str,
    filename: str,
    *,
    prompt_id: str,
    prompt_source: str,
    parent_prompt_id: str | None = None,
) -> None:
    """
    Minimal MVP: attach prompt linkage fields to the per-image metadata record.
    """
    meta = _load_image_metadata(slug)
    if filename not in meta or not isinstance(meta[filename], dict):
        meta[filename] = {}

    meta[filename]["prompt_id"] = prompt_id
    meta[filename]["prompt_source"] = prompt_source
    meta[filename]["parent_prompt_id"] = parent_prompt_id

    # NOTE: If your project uses a different save function name, replace this call.
    _save_image_metadata(slug, meta)


# Only place prompt-based generation is done
def _render_prompt_generation_ui(slug: str) -> None:
    st.markdown("### 🎨 Generate Images From Prompt")

    # -------------------------
    # GenStudio: Creative Intent (human-friendly)
    # -------------------------
    st.markdown("#### Describe what you want")

    intent_text = (
        st.text_area(
            "Write freely — goals, mood, audience, constraints. You can paste briefs, notes, or copy.",
            value=st.session_state.get("genstudio_intent_text", ""),
            height=180,
            key="genstudio_intent_text",
        )
        or ""
    ).strip()

    with st.expander("Open generator", expanded=False):

        # -------------------------
        # Prompt refinement toggle (default ON)
        # -------------------------
        refine_enabled = st.checkbox(
            "Refine prompt for best results",
            value=st.session_state.get("genstudio_refine_enabled", True),
            key="genstudio_refine_enabled",
            help="When enabled, your free-form intent will be refined into a high-quality generation prompt.",
        )

        # TEMP override (optional)
        override = (
            st.text_area(
                "Legacy prompt (temporary override)",
                placeholder="Optional: override the intent text for generation…",
                height=120,
                key=f"gen_override_{slug}",
            )
            or ""
        ).strip()

        # -------------------------
        # Determine text used for generation
        # -------------------------
        override_clean = (override or "").strip()
        used_prompt = override_clean if override_clean else (intent_text or "").strip()

        refined_prompt = used_prompt
        refinement = None

        if refine_enabled and (not override_clean) and used_prompt:
            try:
                client = OpenAI()
                refinement = refine_prompt(used_prompt, client=client)
                if refinement and getattr(refinement, "refined_prompt", None):
                    refined_prompt = (refinement.refined_prompt or "").strip()
            except Exception as e:  # noqa: BLE001
                st.warning(f"Prompt refinement failed; using original text. ({e})")
                refined_prompt = used_prompt
                refinement = None

        # ✅ Safety fallback: never allow empty refined_prompt
        if not (refined_prompt or "").strip():
            refined_prompt = used_prompt

        # Clean versions used everywhere downstream
        used_prompt_clean = (used_prompt or "").strip()
        refined_prompt_clean = (refined_prompt or "").strip() or used_prompt_clean

        st.caption(
            f"Refine={refine_enabled} | Override_set={bool(override_clean)} | changed={refined_prompt_clean != used_prompt_clean}"
        )

        # -------------------------
        # Controls
        # -------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            engine = st.selectbox(
                "Engine",
                ["stability", "openai", "nanobanana"],
                index=0,
                help="Which image engine to use.",
                key=f"gen_engine_{slug}",
            )

        with col2:
            n_images = st.slider(
                "Number of images",
                min_value=1,
                max_value=6,
                value=2,
                key=f"gen_n_{slug}",
            )

        with col3:
            size = st.selectbox(
                "Size (ignored by some engines)",
                ["1024x1024", "768x768"],
                index=0,
                key=f"gen_size_{slug}",
            )

        # -------------------------
        # Preview
        # -------------------------
        with st.expander("Preview text to be generated", expanded=False):
            st.write("This is the text that will be sent to the image engine.")

            if refinement and refined_prompt_clean and refined_prompt_clean != used_prompt_clean:
                st.caption("Refined prompt (used for generation):")
                st.code(refined_prompt_clean, language="text")
                st.caption("Original intent:")
                st.code(used_prompt_clean or "(empty)", language="text")
            else:
                st.caption("Prompt (used for generation):")
                st.code(refined_prompt_clean or "(empty)", language="text")

        # -------------------------
        # Action (outside preview expander)
        # -------------------------
        if st.button("Generate images", type="primary", key=f"gen_btn_{slug}"):

            # Guard: must have something to generate
            if not used_prompt_clean:
                st.warning("Please enter your description (or provide an override).")
                return

            # =========================================================
            # ✅ Intent -> refined lineage (Step 4)
            # =========================================================
            refinement_used = bool(
                refine_enabled
                and refined_prompt_clean
                and refined_prompt_clean != used_prompt_clean
                and not override_clean  # if override is set, treat as manual
            )

            intent_prompt_id: str | None = None

            if refinement_used:
                # 1) Upsert INTENT prompt
                intent_prompt_id = upsert_prompt_record(
                    slug,
                    prompt_text=used_prompt_clean,
                    input_text=intent_text,
                    source="manual",
                    parent_prompt_id=None,
                )

                # 2) Upsert REFINED prompt linked to intent
                prompt_source = "refine"
                prompt_id = upsert_prompt_record(
                    slug,
                    prompt_text=refined_prompt_clean,
                    input_text=intent_text,
                    source=prompt_source,
                    parent_prompt_id=intent_prompt_id,
                )
            else:
                prompt_source = "manual"
                prompt_id = upsert_prompt_record(
                    slug,
                    prompt_text=refined_prompt_clean,
                    input_text=intent_text,
                    source=prompt_source,
                    parent_prompt_id=None,
                )

            # -------------------------
            # Generate + Save
            # -------------------------
            with st.spinner(f"Generating {n_images} image(s) with {engine}…"):
                try:
                    img_bytes_list = _generate_images_for_engine(
                        engine,
                        refined_prompt_clean,
                        n_images,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                    return

                saved_paths: list[Path] = []
                failed: list[str] = []

                for idx, img_bytes in enumerate(img_bytes_list, start=1):
                    try:
                        path, asset_id = _save_image_bytes(  # noqa: F821
                            slug,
                            img_bytes,
                            generated=True,
                            kind="origin",
                            engine=engine,
                            prompt_text=refined_prompt_clean,
                            input_text=intent_text,
                            refine_enabled=refine_enabled,
                            size=size,
                        )
                        saved_paths.append(path)

                        _attach_prompt_to_image_metadata(
                            slug,
                            path.name,
                            prompt_id=prompt_id,
                            prompt_source=prompt_source,
                            parent_prompt_id=intent_prompt_id,
                        )

                    except TypeError:
                        # Backward-compat: if _save_image_bytes doesn't accept the new kwargs yet
                        path, asset_id = _save_image_bytes(  # noqa: F821
                            slug,
                            img_bytes,
                            generated=True,
                            kind="origin",
                            engine=engine,
                            prompt_text=refined_prompt_clean,
                        )
                        saved_paths.append(path)

                        _attach_prompt_to_image_metadata(
                            slug,
                            path.name,
                            prompt_id=prompt_id,
                            prompt_source=prompt_source,
                            parent_prompt_id=intent_prompt_id,
                        )

                    except Exception as e:  # noqa: BLE001
                        failed.append(f"Image {idx}: {e}")

            if saved_paths:
                st.success(f"Saved {len(saved_paths)} image(s) to this campaign.")
            if failed:
                st.warning("Some images failed to save:")
                for msg in failed[:8]:
                    st.write(f"• {msg}")
                if len(failed) > 8:
                    st.write(f"• ...and {len(failed) - 8} more")

            st.rerun()

def _render_export_section(slug: str) -> None:
    st.markdown("### 📦 Export Campaign Images")

    images = _list_all_images(slug)
    if not images:
        st.info("No images to export yet. Generate or upload images first.")
        return

    st.write(f"This campaign currently has **{len(images)}** image(s).")

    if st.button("Prepare ZIP file"):
        with st.spinner("Building ZIP of campaign images…"):
            zip_bytes = _build_images_zip(slug)

        if not zip_bytes:
            st.warning("No images were found to export.")
            return

        st.success("ZIP file ready. Click below to download:")
        st.download_button(
            label="⬇️ Download campaign_images.zip",
            data=zip_bytes,
            file_name=f"{slug}_images.zip",
            mime="application/zip",
        )

def _gallery_caption(filename: str, meta: dict, info_override: dict | None = None) -> str:
    info = info_override or meta.get(filename, {})
    if not isinstance(info, dict):
        info = {}

    legacy = meta.get(filename, {})
    if not isinstance(legacy, dict):
        legacy = {}

    favorite = bool(legacy.get("favorite"))
    selected = bool(legacy.get("selected"))

    prefix = ""
    if favorite and selected:
        prefix = "⭐✔︎ "
    elif favorite:
        prefix = "⭐ "
    elif selected:
        prefix = "✔︎ "

    kind = (info.get("kind") or "").strip().lower()
    kind_label = "Variant" if kind == "variant" else "Original"

    # Version / current (use legacy because those are UI/workflow fields)
    v = legacy.get("version")
    try:
        v_int = int(v) if v is not None else None
    except Exception:
        v_int = None

    parts = [kind_label]
    if v_int is not None:
        parts.append(f"v{v_int}")

    if legacy.get("is_current") is True:
        parts.append("Current")

    # Derived-from: base_image likely comes from index (info), but base version is in legacy meta
    base_name = info.get("base_image")
    if base_name and isinstance(meta.get(base_name), dict):
        base_info = meta[base_name]
        base_v = base_info.get("version")
        try:
            base_v_int = int(base_v) if base_v is not None else None
        except Exception:
            base_v_int = None

        if base_v_int is not None:
            parts.append(f"derived from v{base_v_int}")
        else:
            parts.append("derived from base")

    return prefix + " · ".join(parts)


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem  # e.g. "1765562105678" or "evo_orange_hero"
    if stem.isdigit():
        return f"Image {stem[-6:]}"
    pretty = stem.replace("_", " ").replace("-", " ").strip()
    return pretty.title()

def _render_gallery(slug: str) -> None:
    st.markdown("### 🖼️ Campaign Image Library")
    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = []
    all_images = _list_all_images(slug)
    meta = _load_image_metadata(slug)
    meta, ensured = _ensure_image_metadata_schema(meta)

    # --- NEW: Load canonical asset index (read-only enrichment) ---
    assets_index = load_assets_index(slug)  # asset_id -> record
    index_by_filename = {
        rec.get("filename"): rec for rec in assets_index.values()
        if isinstance(rec, dict) and rec.get("filename")
    }

    def merged_info(filename: str) -> dict:
        """
        Combine canonical index info + legacy per-filename meta.
        Canonical fields come from assets_index; UI flags come from legacy meta.
        Legacy meta wins on conflicts so favorites/selected/etc stay editable.
        """
        idx = index_by_filename.get(filename, {})
        legacy = meta.get(filename, {})
        if not isinstance(legacy, dict):
            legacy = {}
        if not isinstance(idx, dict):
            idx = {}
        return {**idx, **legacy}
    
    def family_key_for(p: Path) -> str:
        info = merged_info(p.name)
        return str(info.get("root_id") or info.get("asset_id") or p.name)

    # ---------- NEW: family sorting helpers ----------

    def _version_int(p: Path) -> int:
        info = merged_info(p.name)
        try:
            return int(info.get("version") or 0)
        except Exception:
            return 0

    def _is_current(p: Path) -> bool:
        return bool(merged_info(p.name).get("is_current") is True)

    def _created_at(p: Path) -> str:
        return str(merged_info(p.name).get("created_at") or "")

    def sort_family_paths(fam_paths: list[Path]) -> list[Path]:
        """
        Order within a family:
          1) Current first
          2) Higher version first
          3) Newer created_at first (fallback to file mtime)
          4) Filename as deterministic fallback
        """
        # Pass 1 (stable): newest-first by created_at, fallback to mtime, then name
        tmp = sorted(
            fam_paths,
            key=lambda p: (
                _created_at(p) or "",
                p.stat().st_mtime,
                p.name,
            ),
            reverse=True,
        )

        # Pass 2 (stable): current first, then version desc
        return sorted(
            tmp,
            key=lambda p: (
                0 if _is_current(p) else 1,
                -_version_int(p),
            ),
        )


    if ensured:
        _save_image_metadata(slug, meta)

    if not all_images:
        st.info("No images yet. Generate or upload images first.")
        return
    
    # ---------- Filters ----------
    f1, f2, f3, f4 = st.columns([1.2, 1.6, 2.2, 1.6])

    with f1:
        show_only_favorites = st.checkbox("Show only favorites", value=False)

    with f2:
        show_only_current = st.checkbox("Show only current versions", value=False)

    with f3:
        type_options = ["All", "hero", "lifestyle", "product_only", "background", "social", "supporting"]
        selected_type = st.selectbox("Filter by type", type_options, index=0)

    with f4:
        group_by_family = st.checkbox("Group by asset family", value=True)


    images = all_images

    if show_only_favorites:
        images = [p for p in images if bool(meta.get(p.name, {}).get("favorite"))]

    if show_only_current:
        images = [p for p in images if meta.get(p.name, {}).get("is_current") is True]

    if selected_type != "All":
        images = [p for p in images if meta.get(p.name, {}).get("asset_type") == selected_type]

    if not images:
        st.info("No images match this filter yet.")
        return


    # ---------- Preview + Info Panels ----------
    preview_key = f"{slug}_preview_image"
    info_key = f"{slug}_info_image"

    preview_path = st.session_state.get(preview_key)
    info_name = st.session_state.get(info_key)

    if preview_path:
        st.markdown("#### 🔍 Preview")
        st.image(preview_path, use_container_width=True)
        if st.button("Close preview"):
            st.session_state[preview_key] = None
            st.rerun()
        st.markdown("---")

        info = merged_info(info_name) if info_name else {}

        if info_name and info:
            st.markdown("#### ℹ️ Image info")
            st.write(f"**File:** `{info_name}`")

            if "kind" in info:
                st.write(f"**Kind:** {info['kind']}")
            if "engine" in info:
                st.write(f"**Engine:** {info['engine']}")

            if "prompt" in info:
                with st.expander("Prompt", expanded=False):
                    st.write(info["prompt"])

            if "instructions" in info:
                with st.expander("Instructions", expanded=False):
                    st.write(info["instructions"])

            if "base_image" in info:
                st.write(f"**Base image:** `{info['base_image']}`")

            if "original_filename" in info:
                st.write(f"**Original filename:** `{info['original_filename']}`")

            if "created_at" in info:
                st.write(f"**Created at:** {info['created_at']} (UTC)")

            if info.get("parent_id"):
                st.write(f"**Parent asset:** `{info['parent_id']}`")
            if info.get("root_id"):
                st.write(f"**Root asset:** `{info['root_id']}`")

            if st.button("Close info"):
                st.session_state[info_key] = None
                st.rerun()

            st.markdown("---")


        st.markdown("#### ℹ️ Image info")
        st.write(f"**File:** `{info_name}`")

        if "kind" in info:
            st.write(f"**Kind:** {info['kind']}")
        if "engine" in info:
            st.write(f"**Engine:** {info['engine']}")

        if "prompt" in info:
            with st.expander("Prompt", expanded=False):
                st.write(info["prompt"])

        if "instructions" in info:
            with st.expander("Instructions", expanded=False):
                st.write(info["instructions"])

        if "base_image" in info:
            st.write(f"**Base image:** `{info['base_image']}`")

        if "original_filename" in info:
            st.write(f"**Original filename:** `{info['original_filename']}`")

        if "created_at" in info:
            st.write(f"**Created at:** {info['created_at']} (UTC)")

        if st.button("Close info", key=f"close_info_{slug}_{st.session_state.get('selected_image_name','none')}"):
            st.session_state["selected_image_name"] = None
            st.rerun()


        st.markdown("---")

    # ---------- Sorting ----------
    # Do NOT sort by "selected" (keeps layout stable)
    def sort_key(p: Path):
        info = meta.get(p.name, {})
        favorite = bool(info.get("favorite"))
        return (0 if favorite else 1, -p.stat().st_mtime)

    images_sorted = sorted(images, key=sort_key)
    
    # ---------- CSS for centered image thumbnails ----------
    st.markdown("""
<style>
.caf-thumb-box {
    border-radius: 8px;
    border: 1px solid #ddd;
    overflow: hidden;
    width: 200px;
    height: 200px;
    margin: 0 auto 0.25rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fafafa;
}
.caf-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
}

/* Selected thumbnail highlight */
.caf-thumb-box.selected {
    outline: 3px solid rgba(59, 130, 246, 0.9);
    outline-offset: 2px;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

.caf-link-row button {
    border: none;
    background: none;
    padding: 0 4px;
    margin: 0;
    font-size: 0.9rem;
}

.favorite-badge {
    position: absolute;
    top: 4px;
    right: 6px;
    background: rgba(255, 221, 0, 0.85);
    padding: 2px 6px;
    border-radius: 6px;
}

.image-card {
    padding-bottom: 10px;
}

.image-wrapper {
    position: relative; /* needed for overlays */
}

/* Label bar ABOVE the image (replaces overlay bubble) */
.caf-thumb-label {
    position: static;              /* <-- critical */
    margin: 0 auto 6px auto;
    width: 200px;                  /* match thumb */
    padding: 4px 8px;
    border-radius: 8px;
    font-size: 0.85rem;
    line-height: 1.1;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(0, 0, 0, 0.12);
}

.caf-thumb-label .big {
    font-weight: 700;
}

.caf-thumb-label.current {
    border: 1px solid rgba(34, 197, 94, 0.45);
}
</style>

""", unsafe_allow_html=True)
    
    selection_states = {}
    meta_changed = False

    # ---------- Main layout: gallery (left) + inspector (right) ----------
    gallery_col, inspector_col = st.columns([5, 2], gap="large")
    with gallery_col:

        def render_grid(image_paths: list[Path]) -> None:
            cols = st.columns(4)
            for idx, img_path in enumerate(image_paths):
                col = cols[idx % 4]
                with col:
                    legacy = meta.get(img_path.name, {})
                    if not isinstance(legacy, dict):
                        legacy = {}

                    info = merged_info(img_path.name)  # enriched
                    caption = _gallery_caption(img_path.name, meta, info_override=info)

                    favorite = bool(legacy.get("favorite"))
                    selected = bool(legacy.get("selected"))

                    # ---- Card wrapper ----
                   # st.markdown('<div class="image-card">', unsafe_allow_html=True)
                   # st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)

                    # ---- Image thumbnail ----
                    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                    is_active = st.session_state.get("selected_image_name") == img_path.name
                    selected_cls = " selected" if is_active else ""

                    # Build label text from merged info
                    v = info.get("version")
                    kind = info.get("kind", "origin")
                    is_current = info.get("is_current") is True

                    if kind == "origin":
                        label_text = f"V{v} · Original" if v else "Original"
                    elif is_current:
                        label_text = f"V{v} · Current"
                    else:
                        label_text = f"V{v} · Variant" if v else "Variant"

                    label_cls = "caf-thumb-label current" if is_current else "caf-thumb-label"

                    html = (
                        f'<div class="caf-thumb-card">'
                        f'<div class="{label_cls}">'
                        f'<span class="big">{label_text}</span>'
                        f'</div>'
                        f'<div class="caf-thumb-box{selected_cls}">'
                        f'<img src="data:image/png;base64,{b64}">'
                        f'</div>'
                        f'</div>'
                    )

                    st.markdown(html, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)  # close image-wrapper

                    # ---------- Action Icons ----------
                    lc1, lc2, lc3 = st.columns(3)

                    with lc1:
                        if st.button("🔍", key=f"view_{slug}_{img_path.name}"):
                            st.session_state["selected_image_name"] = img_path.name
                            st.session_state[preview_key] = str(img_path)
                            st.rerun()

                    with lc2:
                        star_label = "⭐" if favorite else "☆"
                        if st.button(star_label, key=f"fav_{slug}_{img_path.name}"):
                            legacy["favorite"] = not favorite
                            meta[img_path.name] = legacy
                            nonlocal_meta_changed[0] = True

                    with lc3:
                        if st.button("ℹ️", key=f"info_{slug}_{img_path.name}"):
                            st.session_state["selected_image_name"] = img_path.name
                            st.session_state[info_key] = img_path.name
                            st.rerun()

                    # ---------- Select Checkbox ----------
                    sel_key = f"sel_{slug}_{img_path.name}"
                    sel_value = st.checkbox("Select", key=sel_key, value=selected)
                    selection_states[img_path.name] = sel_value

                    st.caption(caption)

                    st.markdown('</div>', unsafe_allow_html=True)  # close image-card

        # allow nested function to mark meta_changed
        nonlocal_meta_changed = [False]

        if group_by_family:
            families: dict[str, list[Path]] = {}
            for p in images_sorted:
                fk = family_key_for(p)
                families.setdefault(fk, []).append(p)

            for fk, fam_paths in families.items():
                fam_paths = sort_family_paths(fam_paths)

                # Pick a lead for the header: current > origin > first
                lead = None
                for p in fam_paths:
                    if bool(merged_info(p.name).get("is_current") is True):
                        lead = p
                        break

                if lead is None:
                    for p in fam_paths:
                        if merged_info(p.name).get("kind") == "origin":
                            lead = p
                            break
                if lead is None:
                    lead = fam_paths[0]

                st.markdown(f"#### {_title_from_filename(lead.name)}")
                render_grid(fam_paths)
                st.markdown("---")
        else:
            render_grid(images_sorted)

        meta_changed = meta_changed or nonlocal_meta_changed[0]
        st.session_state["selected_images"] = [fn for fn, sel in selection_states.items() if sel]

 
    with inspector_col:
        st.subheader("Inspector")

        selected_name = st.session_state.get("selected_image_name")

        if not selected_name:
            st.caption("Click ℹ️ on an image to view/edit metadata.")
        else:
            info = merged_info(selected_name)

            # ---------- Auto-default metadata ----------
            info_changed = False

            if "title" not in info or not str(info.get("title", "")).strip():
                info["title"] = _title_from_filename(selected_name)
                info_changed = True

            valid_types = ["hero", "lifestyle", "product_only", "background", "social", "supporting"]
            if "asset_type" not in info or info.get("asset_type") not in valid_types:
                first_name = images_sorted[0].name if images_sorted else ""
                info["asset_type"] = "hero" if selected_name == first_name else "supporting"
                info_changed = True

            if "channels" not in info or not isinstance(info.get("channels"), list):
                info["channels"] = []
                info_changed = True

            if "approval_status" not in info or info.get("approval_status") not in ["draft", "approved", "rejected"]:
                info["approval_status"] = "draft"
                info_changed = True

            if "usage_rights" not in info or info.get("usage_rights") not in ["internal", "external", "paid_media", "unrestricted"]:
                info["usage_rights"] = "internal"
                info_changed = True

            if "notes" not in info:
                info["notes"] = ""
                info_changed = True

            if info_changed:
                meta[selected_name] = info
                meta_changed = True


            # ---------- Image preview ----------
            selected_path = next((p for p in images_sorted if p.name == selected_name), None)
            if selected_path and selected_path.exists():
                st.image(str(selected_path), use_container_width=True)
            # ---------- Versioning actions (MVP) ----------
            st.markdown("### Versioning (actions)")

            root_id = info.get("root_id") or info.get("asset_id")

            cA, cB = st.columns([1, 1])

            with cA:
                if st.button("✅ Make Current Version", key=f"make_current_{slug}_{selected_name}"):
                    # Reload fresh metadata to avoid stale info
                    meta2 = _load_image_metadata(slug)

                    # Determine root_id from selected file (fresh)
                    sel_info = meta2.get(selected_name, {}) if isinstance(meta2.get(selected_name, {}), dict) else {}
                    sel_root = sel_info.get("root_id") or sel_info.get("asset_id")
                    if not sel_root:
                        st.error("Selected image is missing asset_id/root_id; cannot version.")
                        if not sel_root:
                            st.error("Selected image is missing asset_id/root_id; cannot version.")
                            st.stop()
                    # Flip all in same lineage to not-current
                    changed_any = False
                    for fn, inf in meta2.items():
                        if not isinstance(inf, dict):
                            continue
                        rid = inf.get("root_id") or inf.get("asset_id")
                        if rid == sel_root and inf.get("is_current") is True:
                            inf["is_current"] = False
                            meta2[fn] = inf
                            changed_any = True

                    # Mark selected as current
                    if isinstance(sel_info, dict):
                        if "version" not in sel_info:
                            sel_info["version"] = 1
                        sel_info["is_current"] = True
                        meta2[selected_name] = sel_info
                        changed_any = True

                    if changed_any:
                        _save_image_metadata(slug, meta2)
                        st.success("Marked as current.")
                        st.rerun()

            with cB:
                st.caption(f"Root: {str(root_id)[:8] if root_id else ''}")


            # ---------- Provenance (read-only) ----------
            engine_val = str(info.get("engine", ""))
            input_text_val = str(info.get("input_text") or "")
            prompt_text_val = str(info.get("prompt_text") or info.get("prompt") or info.get("instructions") or "")


            st.markdown("### Provenance (read-only)")
            st.text_input(
                "Filename",
                value=selected_name,
                disabled=True,
                key=f"ins_fn_{slug}_{selected_name}",
            )
            st.text_input(
                "Engine",
                value=engine_val,
                disabled=True,
                key=f"ins_engine_{slug}_{selected_name}",
            )

            # Creative intent (original)
            st.text_area(
                "Creative intent (original)",
                value=input_text_val,
                disabled=True,
                height=120,
                key=f"ins_intent_{slug}_{selected_name}",
            )

            # Refined prompt used
            st.text_area(
                "Refined prompt (used)",
                value=prompt_text_val,
                disabled=True,
                height=180,
                key=f"ins_prompt_{slug}_{selected_name}",
            )

            # ---------- Lineage & Versioning (read-only) ----------
            st.markdown("### Lineage & Versioning (read-only)")

            # Clean read-only fields
            st.text_input("Kind", value=str(info.get("kind", "")), disabled=True)
            st.text_input("Asset ID", value=str(info.get("asset_id", "")), disabled=True)
            st.text_input("Root ID", value=str(info.get("root_id", "")), disabled=True)
            st.text_input("Parent ID", value=str(info.get("parent_id", "")), disabled=True)
            st.text_input("Base image", value=str(info.get("base_image", "")), disabled=True)
            st.text_input(
                "Derivation",
                value=str((info.get("derivation") or {}).get("type", "")),
                disabled=True,
            )
            st.text_input("Version", value=str(info.get("version", "")), disabled=True)
            st.text_input("Current", value=str(info.get("is_current", "")), disabled=True)
            st.text_input("Supersedes", value=str(info.get("supersedes_id", "")), disabled=True)
            st.text_input("Created at", value=str(info.get("created_at", "")), disabled=True)
            st.text_input("Status", value=str(info.get("status", "")), disabled=True)

            # Raw JSON (hidden unless needed)
            with st.expander("Show raw lineage JSON", expanded=False):
                st.json({
                    "kind": info.get("kind"),
                    "asset_id": info.get("asset_id"),
                    "root_id": info.get("root_id"),
                    "parent_id": info.get("parent_id"),
                    "base_image": info.get("base_image"),
                    "derivation": info.get("derivation"),
                    "version": info.get("version"),
                    "is_current": info.get("is_current"),
                    "supersedes_id": info.get("supersedes_id"),
                    "created_at": info.get("created_at"),
                    "status": info.get("status"),
                })


            # ---------- Metadata (editable) ----------
            # (your existing editable fields here)
            st.markdown("### Metadata (editable)")

            title = st.text_input(
                "Title",
                value=str(info.get("title", "")),
                key=f"ins_title_{slug}_{selected_name}",
            )

            type_options = ["hero", "lifestyle", "product_only", "background", "social", "supporting"]
            current_type = info.get("asset_type", "supporting")
            if current_type not in type_options:
                current_type = "supporting"

            asset_type = st.selectbox(
                "Type",
                type_options,
                index=type_options.index(current_type),
                key=f"ins_type_{slug}_{selected_name}",
            )

            channels = st.multiselect(
                "Channels",
                ["social", "web", "email", "print", "ads"],
                default=info.get("channels", []),
                key=f"ins_channels_{slug}_{selected_name}",
            )

            approval_options = ["draft", "approved", "rejected"]
            approval_status = st.selectbox(
                "Approval status",
                approval_options,
                index=approval_options.index(info.get("approval_status", "draft")),
                key=f"ins_approval_{slug}_{selected_name}",
            )

            rights_options = ["internal", "external", "paid_media", "unrestricted"]
            usage_rights = st.selectbox(
                "Usage rights",
                rights_options,
                index=rights_options.index(info.get("usage_rights", "internal")),
                key=f"ins_rights_{slug}_{selected_name}",
            )

            notes = st.text_area(
                "Notes",
                value=str(info.get("notes", "")),
                height=120,
                key=f"ins_notes_{slug}_{selected_name}",
            )
            # second save button:
            c1, c2 = st.columns([1, 1])

            with c1:
                if st.button("Save metadata", key=f"ins_save_{slug}_{selected_name}"):
                    info["title"] = title
                    info["asset_type"] = asset_type
                    info["channels"] = channels
                    info["approval_status"] = approval_status
                    info["usage_rights"] = usage_rights
                    info["notes"] = notes

                    meta[selected_name] = info
                    meta_changed = True

                    st.success("Metadata saved.")

            with c2:
                if st.button("Close", key=f"ins_close_{slug}_{selected_name}"):
                    st.session_state["selected_image_name"] = None
                    st.rerun()

            # end second save button

    # ---------- Sync selection metadata ----------
    for filename, selected in selection_states.items():
        info = meta.get(filename, {})
        if selected and not info.get("selected"):
            info["selected"] = True
            meta_changed = True
        if not selected and info.get("selected"):
            info["selected"] = False
            meta_changed = True
        meta[filename] = info
    st.session_state["selected_images"] = [
        fn for fn, inf in meta.items()
        if isinstance(inf, dict) and inf.get("selected") is True
]

    # ---------- Bulk Delete ----------
    if st.button("Delete selected images"):
        to_delete = [name for name, sel in selection_states.items() if sel]

        if not to_delete:
            st.info("No images selected.")
        else:
            # Track which lineages lost their current
            affected_roots: set[str] = set()

            # Figure out which deleted items were current (and their root_ids)
            for name in to_delete:
                info = meta.get(name, {})
                if isinstance(info, dict) and info.get("is_current") is True:
                    rid = info.get("root_id") or info.get("asset_id")
                    if rid:
                        affected_roots.add(str(rid))

            # Delete files + metadata entries
            for img_path in images_sorted:
                if img_path.name in to_delete:
                    try:
                        img_path.unlink()
                    except FileNotFoundError:
                        pass
                    meta.pop(img_path.name, None)

            # Auto-promote a new current in any affected lineage
            if affected_roots:
                meta = _auto_promote_current_after_delete(meta, affected_roots)

            _save_image_metadata(slug, meta)
            st.success(f"Deleted {len(to_delete)} image(s).")
            st.rerun()


    # Save metadata after favorites or selection changes
    if meta_changed:
        _save_image_metadata(slug, meta)
        st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Image Library")

    slug = _get_current_slug()
    if not slug:
        st.warning("No campaign selected. Please choose a campaign on the Dashboard first.")
        return

    #temporary insert
    from caf_app.prompt_store import (
        campaign_prompts_path,
        load_prompts_index,
        upsert_prompt_record,
    )

    st.markdown("### 🧪 Prompt Library Sanity Test (temporary)")

    slug = _get_current_slug()
    st.write("slug =", slug)

    if slug:
        st.write("prompts_index path =", str(campaign_prompts_path(slug)))

        if st.button("Create test prompt record"):
            pid = upsert_prompt_record(
                slug,
                prompt_text="TEST PROMPT: banana mango hero image",
                input_text="test input",
                source="manual",
                parent_prompt_id=None,
            )
            data = load_prompts_index(slug)
            st.success(f"Created/updated prompt_id: {pid}")
            st.write("prompt records =", len(data.get("prompts", {})))
    else:
        st.warning("No campaign selected (slug is empty).")

    #end temporary insert

    # Inspector state (right panel selection)
    if "selected_image_name" not in st.session_state:
        st.session_state["selected_image_name"] = None

    _render_campaign_header(slug)

    st.divider()
    _render_tools_panel(slug)

    st.divider()
    _render_gallery(slug)


if __name__ == "__main__":
    main()
