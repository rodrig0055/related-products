"""Review page for one run: every PDP with its 4 proposed cards, why each was picked, and the totals.

  uv run python -m relprod.report out/demo/<date>
  uv run python -m relprod.report out/demo/<date> --live   # rollout review: diff against the A/B test sets

--live compares every PDP with what the A/B test put live (out/<shop>/abtest/treatment_sets.json,
verified equal to product_related on upload): engine-arm sets kept or changed, rotation PDPs new.

Thumbnails are downloaded once into cache/<shop>/thumbs/ (112px WebP via cwebp) and embedded as
data URIs, so the report is one self-contained file.
"""

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .shops import ROOT, SHOPS

TEMPLATE = Path(__file__).with_name("report_template.html")


def thumbs(shop_key: str, products: dict) -> dict[str, str]:
    folder = ROOT / "cache" / shop_key / "thumbs"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for pid, p in products.items():
        f = folder / f"{pid}.webp"
        if not f.exists() and p.get("image") and shutil.which("cwebp"):
            src = folder / f"{pid}.src"
            subprocess.run(["curl", "-s", "-m", "20", "-o", str(src), p["image"]], check=False)
            if src.exists():
                subprocess.run(["cwebp", "-quiet", "-resize", "112", "0", "-q", "70", str(src), "-o", str(f)], check=False)
                src.unlink()
        if f.exists():
            out[pid] = "data:image/webp;base64," + base64.b64encode(f.read_bytes()).decode()
    return out


def build(run_dir: Path, live: bool = False) -> Path:
    data = json.loads((run_dir / "data.json").read_text())
    shop_key = run_dir.parent.name
    shop = SHOPS[shop_key]
    data["shop"] = {"name": shop.name, "origin": shop.origin, "currency": shop.currency}
    effect = run_dir / "tipr_effect.json"
    data["tipr"] = json.loads(effect.read_text()) if effect.exists() else None
    data["live"] = None
    if live:
        kit = run_dir.parent / "abtest"
        sets = json.loads((kit / "treatment_sets.json").read_text())["sets"]
        data["live"] = {str(k): [int(x) for x in v] for k, v in sets.items()}
    imgs = thumbs(shop_key, data["products"])
    html = (TEMPLATE.read_text()
            .replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
            .replace("/*IMGS*/null", json.dumps(imgs, separators=(",", ":"))))
    out = run_dir / "report.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    path = build(Path(sys.argv[1]).resolve(), "--live" in sys.argv[2:])
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")
