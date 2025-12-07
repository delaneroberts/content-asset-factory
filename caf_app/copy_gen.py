from pathlib import Path
import json
from datetime import datetime

from openai import OpenAI
from config.settings import OPENAI_API_KEY

# Directory where the raw copy files will be stored
RAW_COPY_DIR = Path("assets/copy/raw")
RAW_COPY_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)


def _fallback_copy(brief: str) -> dict:
    """
    Safe fallback if the OpenAI call or JSON parsing fails.
    Returns the same structure as the main function.
    """
    headlines = [
        "Meet EVO Soda: Bold Flavor, Clean Ingredients",
        "Summer Just Got Lighter with EVO Soda",
        "Fizz Without the Guilt: EVO Soda Arrives",
    ]
    ctas = [
        "Try EVO Soda Today",
        "Find EVO Near You",
        "Taste the New Kind of Soda",
    ]
    raw_text = (
        "Fallback copy for EVO Soda campaign.\n\n"
        "Headlines:\n- " + "\n- ".join(headlines) +
        "\n\nCTAs:\n- " + "\n- ".join(ctas) +
        f"\n\nBrief used:\n{brief}"
    )

    return {
        "headlines": headlines,
        "ctas": ctas,
        "raw_text": raw_text,
    }

def generate_evo_copy(brief: str) -> dict:
    """
    TEMP VERSION for demo stability:
    Skip the OpenAI call and always return a solid fallback
    so the app never hangs.
    """
    return _fallback_copy(brief)

def save_copy_to_file(campaign_name: str, copy_data: dict) -> Path:
    """
    Save the raw_text portion of copy_data to a timestamped file
    under assets/copy/raw and return the file path.
    """
    raw_text = copy_data.get("raw_text", "")

    safe_name = campaign_name.strip().lower().replace(" ", "_") or "campaign"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_copy_{timestamp}.txt"

    out_path = RAW_COPY_DIR / filename
    out_path.write_text(raw_text, encoding="utf-8")
    return out_path
