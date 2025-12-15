# tools/rebuild_assets_index.py
from pathlib import Path
import sys

from caf_app.asset_store import AssetStore

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/rebuild_assets_index.py <campaign_slug>")
        return 2

    slug = sys.argv[1]
    campaigns_root = Path("campaigns")  # adjust if your root differs
    store = AssetStore(campaigns_root=campaigns_root)

    n, skipped = store.rebuild_index(slug)
    print(f"Rebuilt index for '{slug}': {n} assets ({skipped} skipped)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
