# tools/rebuild_assets_index.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from caf_app.asset_store import rebuild_assets_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Reindex a CAF campaign's assets_index.json (MVP).")
    ap.add_argument("--campaign", "-c", required=True, help="Campaign slug")
    ap.add_argument(
        "--campaigns-root",
        default="campaigns",
        help="Campaigns root folder (default: campaigns)",
    )
    args = ap.parse_args()

    campaigns_root = Path(args.campaigns_root)
    out_path = campaigns_root / args.campaign / "assets_index.json"

    try:
        warnings = rebuild_assets_index(args.campaign, campaigns_root=campaigns_root)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print(f"Wrote: {out_path}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f" - {w}")


if __name__ == "__main__":
    main()

