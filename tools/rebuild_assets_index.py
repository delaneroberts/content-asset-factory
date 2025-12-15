# tools/rebuild_assets_index.py
from __future__ import annotations

import argparse
from caf_app.asset_store import assets_index_path, rebuild_assets_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Reindex a CAF campaign's assets_index.json (MVP).")
    ap.add_argument("--campaign", "-c", required=True, help="Campaign slug")
    args = ap.parse_args()

    warnings = rebuild_assets_index(args.campaign)
    print(f"Wrote: {assets_index_path(args.campaign)}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f" - {w}")


if __name__ == "__main__":
    main()
