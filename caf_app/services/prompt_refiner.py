# caf_app/services/prompt_refiner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass(frozen=True)
class PromptRefinement:
    refined_prompt: str
    negative_prompt: Optional[str] = None
    model: str = "gpt-4o-mini"
    version: str = "v1"


_REFINE_SYSTEM = """You are GenStudio Prompt Refiner.
Turn a user's free-form creative intent into a single, high-quality image generation prompt.

Rules:
- Output MUST be a single prompt string, not a list, not JSON.
- Keep it specific: subject, setting, composition, lighting, camera/style cues, mood, constraints.
- Do NOT include policy/disallowed content.
- Do NOT include "negative prompt:" text; return only the positive prompt.
- Keep it under ~900 characters unless the intent truly requires more.
"""


def refine_prompt(intent_text: str, *, client: OpenAI, model: str = "gpt-4o-mini") -> PromptRefinement:
    intent_text = (intent_text or "").strip()
    if not intent_text:
        return PromptRefinement(refined_prompt="", model=model)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,  # stable + repeatable
        messages=[
            {"role": "system", "content": _REFINE_SYSTEM},
            {"role": "user", "content": intent_text},
        ],
    )

    refined = (resp.choices[0].message.content or "").strip()
    return PromptRefinement(refined_prompt=refined, model=model)
