from __future__ import annotations

import os
from typing import Dict

from openai import OpenAI


def _get_client() -> OpenAI | None:
    """
    Return an OpenAI client if OPENAI_API_KEY is set, otherwise None.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _llm_completion(prompt: str, max_tokens: int = 256) -> str:
    """
    Basic LLM completion wrapper.
    Uses a fallback text if no API key is set.
    """
    client = _get_client()
    if client is None:
        # Fallback: simple deterministic text so the app still works
        return f"[FAKE LLM OUTPUT] {prompt[:120]}..."

    # Adjust the model name to whatever you actually have access to
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a marketing copywriter."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_campaign_texts(
    campaign_name: str,
    campaign_brief: str,
) -> Dict[str, str]:
    """
    Generate all text assets for a campaign.
    Returns a dict with keys:
    - tagline
    - slogans
    - value_prop
    - ctas
    - social_posts
    - summary
    """
    base_context = (
        f"Campaign name: {campaign_name}\n\n"
        f"Campaign brief:\n{campaign_brief}\n\n"
        "Target: general consumer audience. Use clear, engaging language."
    )

    prompts = {
        "tagline": (
            f"{base_context}\n\n"
            "Write one short, punchy tagline (max 10 words)."
        ),
        "slogans": (
            f"{base_context}\n\n"
            "Write three alternative slogans, each on its own line."
        ),
        "value_prop": (
            f"{base_context}\n\n"
            "Write a 2–3 sentence value proposition that explains why this "
            "campaign matters to the audience."
        ),
        "ctas": (
            f"{base_context}\n\n"
            "Write three strong call-to-action lines, each on its own line."
        ),
        "social_posts": (
            f"{base_context}\n\n"
            "Write two social media posts:\n"
            "1) A longer post (~100 words) for LinkedIn.\n"
            "2) A short, punchy post (max 30 words) for X/Twitter.\n"
            "Label them clearly as 'LinkedIn:' and 'Twitter:'."
        ),
        "summary": (
            f"{base_context}\n\n"
            "Write a concise paragraph summarizing the campaign concept, tone, "
            "and key message."
        ),
    }

    outputs = {key: _llm_completion(prompt) for key, prompt in prompts.items()}

    # Also return the prompts themselves so we can save them in a text file
    outputs["_prompts"] = "\n\n".join(
        [f"=== {key.upper()} PROMPT ===\n{p}" for key, p in prompts.items()]
    )

    return outputs
