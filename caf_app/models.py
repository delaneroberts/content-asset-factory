# caf_app/models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal, List
from uuid import uuid4

from pydantic import BaseModel, Field


TextAssetKind = Literal["tagline", "header", "subheader", "body"]


class TextAsset(BaseModel):
    """A single piece of campaign copy (tagline, header, body, etc.)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: TextAssetKind
    content: str

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: Literal["ai", "manual"] = "ai"
    prompt: Optional[str] = None
    model_name: Optional[str] = None
    is_favorite: bool = False


class ImageAsset(BaseModel):
    """A generated image associated with a campaign."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    path: str = Field(
        ...,
        description="Filesystem path to the image file, relative to project root.",
    )

    engine: str = Field(..., description="Which generator/engine produced this image.")
    prompt: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_favorite: bool = False


class Concept(BaseModel):
    """
    A concept combines one image with one or more pieces of text
    (tagline, header, subheader, body) for preview/export.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    image_asset_id: str = Field(..., description="ID of the ImageAsset used.")

    # Snapshotted text content (so edits don't break if underlying assets change)
    tagline: Optional[str] = None
    header: Optional[str] = None
    subheader: Optional[str] = None
    body: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_favorite: bool = False


class Campaign(BaseModel):
    """Core data model for a campaign in MVP-1."""

    slug: str = Field(..., description="Filesystem-safe identifier, used as filename")
    name: str = Field(..., description="Human-readable campaign name")

    # Brief fields
    product_name: Optional[str] = None
    description: Optional[str] = None
    audience: Optional[str] = None
    tone: Optional[str] = None
    notes: Optional[str] = None

    # Assets
    text_assets: List[TextAsset] = Field(
        default_factory=list,
        description="All text snippets associated with this campaign.",
    )
    image_assets: List[ImageAsset] = Field(
        default_factory=list,
        description="All generated images associated with this campaign.",
    )
    concepts: List[Concept] = Field(
        default_factory=list,
        description="Built concepts combining images + text.",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }
