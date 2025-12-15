# ui/pages/04_Image_Library.py
from __future__ import annotations

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
from openai import OpenAI, OpenAIError
from huggingface_hub import InferenceClient
from PIL import Image
from caf_app.storage import load_campaign
from caf_app.models import Campaign  # for type hints / future use
from caf_app.asset_store import AssetStore
from caf_app.asset_store import load_assets_index

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
            _render_export_section(slug)


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

    if brief:
        with st.expander("View campaign brief", expanded=False):
            st.write(brief)


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


def _render_prompt_generation_ui(slug: str) -> None:
    st.markdown("### 🎨 Generate Images From Prompt")

    with st.expander("Open generator", expanded=False):
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe the image you want to generate…",
            height=120,
            key=f"gen_prompt_{slug}",
        )

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

        if st.button("Generate images", type="primary", key=f"gen_btn_{slug}"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
                return

            with st.spinner(f"Generating {n_images} image(s) with {engine}…"):
                try:
                    # If your engine function uses size, pass it; otherwise ignore.
                    img_bytes_list = _generate_images_for_engine(engine, prompt.strip(), n_images)
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                    return
            
                versions = _next_versions_for_root(meta, str(base_root_id), len(img_bytes_list))

                saved_paths: list[Path] = []
                #for img_bytes in img_bytes_list:
                for i, img_bytes in enumerate(img_bytes_list):
                    vnum = versions[i]
                    # IMPORTANT: engine must be the selected engine (not hard-coded "openai")
                    for img_bytes in img_bytes_list:
                        path, asset_id = _save_image_bytes(
                            slug,
                            img_bytes,
                            generated=True,
                            kind="origin",
                            engine=engine,
                            prompt=prompt.strip(),
                        )
                    saved_paths.append(path)


                    # No second metadata write here.
                    # _save_image_bytes already:
                    # - upserts AssetStore
                    # - writes legacy metadata including asset_id/family_id

            st.success(f"Saved {len(saved_paths)} image(s) to this campaign.")
            st.rerun()


#Only place variants are generated
def _render_prompt_generation_ui(slug: str) -> None:
    st.markdown("### 🎨 Generate Images From Prompt")

    with st.expander("Open generator", expanded=False):
        prompt = st.text_area(
            "Prompt",
            placeholder="Describe the image you want to generate…",
            height=120,
            key=f"gen_prompt_{slug}",
        )

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

        if st.button("Generate images", type="primary", key=f"gen_btn_{slug}"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
                return

            with st.spinner(f"Generating {n_images} image(s) with {engine}…"):
                try:
                    # If your engine supports size, wire it there.
                    # For now your helper ignores size, so we do too.
                    img_bytes_list = _generate_images_for_engine(engine, prompt.strip(), n_images)
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
                    return

                saved_paths: list[Path] = []
                failed: list[str] = []

                for idx, img_bytes in enumerate(img_bytes_list, start=1):
                    try:
                        # Origin images start NEW families (versioning handled in _save_image_bytes/_update_image_metadata_entry)
                        path, asset_id = _save_image_bytes(
                            slug,
                            img_bytes,
                            generated=True,
                            kind="origin",
                            engine=engine,
                            prompt=prompt.strip(),
                        )
                        saved_paths.append(path)
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

        if info_name:
            info = merged_info(info_name)
            if info:
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

                if "created_at" in info:
                    st.write(f"**Created at:** {info['created_at']} (UTC)")

                # Optional: show lineage IDs if you want
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

        if st.button("Close info"):
            st.session_state[info_key] = None
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

/* ---------- NEW: Version/Current label overlay ---------- */
.caf-thumb-label {
    position: absolute;
    top: 6px;
    left: 6px;
    right: 6px;
    padding: 4px 8px;
    border-radius: 8px;
    font-size: 0.85rem;
    line-height: 1.1;
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(2px);
    z-index: 2;
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
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)

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

                    st.markdown(
                        f'''
                        <div class="image-wrapper">
                            <div class="{label_cls}">
                                <span class="big">{label_text}</span>
                            </div>
                            <div class="caf-thumb-box{selected_cls}">
                                <img src="data:image/png;base64,{b64}">
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )


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

 
    with inspector_col:
        st.subheader("Inspector")

        selected_name = st.session_state.get("selected_image_name")

        if not selected_name:
            st.caption("Click ℹ️ on an image to view/edit metadata.")
        else:
            info = meta.get(selected_name, {})

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
            prompt_val = str(info.get("prompt") or info.get("instructions") or "")

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
            st.text_area(
                "Prompt / Instructions",
                value=prompt_val,
                disabled=True,
                height=120,
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
