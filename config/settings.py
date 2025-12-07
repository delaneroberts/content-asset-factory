from pathlib import Path
import os
from dotenv import load_dotenv

# Point to the .env file in the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def print_config_summary():
    def mask(value: str | None) -> str:
        if not value:
            return "MISSING"
        return value[:4] + "..." + value[-4:]

    print("Config summary:")
    print(f"  OPENAI_API_KEY = {mask(OPENAI_API_KEY)}")

