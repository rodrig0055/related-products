"""Learn from what the cards did, then propose the next sets.

  uv run python -m relprod.learn --shop demo --start 2026-11-03 --end 2026-12-14

1. Exposure per card (A -> B) in the window, per storefront with a GA4 export:
   - a fixed set (product_related rows live in the window): every view of A shows B;
   - the rotation (no rows): A shows 4 random cards from its pool, so B is shown on about
     views(A) x 4 / |pool(A)| of A's views (the same natural experiment uplift.py uses).
2. Clicks and direct sales per card from GA4 (measure.py: a PDP -> PDP click is a related click).
3. Each card's click rate is shrunk toward the average card in its position (empirical Bayes, worth
   PRIOR_EXPOSURES views), so a card shown 30 times with 0 clicks is not condemned yet.
4. factor = shrunk rate / position average, clipped to [FLOOR, CAP]. The engine multiplies a
   card's slot score by it on the next run: cards that earned clicks stay, cards that did not lose
   to relevant cards nobody has tried yet (factor 1). Sales are too sparse per card (a few a month)
   to score cards on, so they are reported, not used; GA4's behaviour pairs keep learning anyway.

Writes cache/<shop>/feedback.json (what the next build uses) and out/<shop>/learn_<end>.json, then
builds the next sets into out/<shop>/<today>/ with the feedback applied.
"""

import argparse
import datetime as dt
import json
from collections import defaultdict

from .measure import click_purchases, flow
from .shops import ROOT, SHOPS
from . import sources

PRIOR_EXPOSURES = 300   # a card's own rate counts as much as the prior after ~300 views of its PDP
FLOOR, CAP = 0.5, 2.0
MIN_EXPOSURES = 50      # below this a card is not scored (factor stays 1)


def card_stats(shop_key: str, start: dt.date, end: dt.date) -> dict:
    from .features import build_products
    from .uplift import rotation_pools
    shop = SHOPS[shop_key]
    cat = sources.catalog(shop)
    products = build_products(shop, cat, {}, {}, {}, {})
    slug_id = {p.slug: p.id for p in products.values()}
    live = sources.current_related(shop) or {}
    pools = rotation_pools(cat, products)
    cards: dict[tuple[int, int], dict] = defaultdict(lambda: dict.fromkeys(
        ("exposures", "clicks", "sales", "revenue"), 0.0))
    position: dict[tuple[int, int], int] = {}
    for front in shop.fronts:
        if not front.ga4_table:
            continue
        fl = flow(front, start, end)
        views = {slug_id[s]: n for s, n in fl["views"].items() if s in slug_id}
        for a, n in views.items():
            if live.get(a):
                for i, b in enumerate(live[a][:4], 1):
                    cards[(a, b)]["exposures"] += n
                    position[(a, b)] = i
            elif pools.get(a):
                share = min(1.0, 4 / len(pools[a]))
                for b in pools[a]:
                    cards[(a, b)]["exposures"] += n * share
        for c in click_purchases(fl, slug_id):
            a, b = slug_id.get(c["src"]), slug_id.get(c["dst"])
            if (a, b) in cards:
                x = cards[(a, b)]
                x["clicks"] += 1
                x["sales"] += c["bought"]
                x["revenue"] += c["revenue"]
    return {"cards": cards, "position": position, "live_sets": len(live)}


def learn(shop_key: str, start: dt.date, end: dt.date, build: bool = True) -> dict:
    st = card_stats(shop_key, start, end)
    cards, position = st["cards"], st["position"]
    # Prior per position for fixed cards (position 1 sits first), one prior for rotation pairs.
    agg = defaultdict(lambda: [0.0, 0.0])
    for k, x in cards.items():
        g = agg[position.get(k, 0)]
        g[0] += x["clicks"]
        g[1] += x["exposures"]
    prior = {pos: (c / e if e else 0.0) for pos, (c, e) in agg.items()}
    fixed_all = [v for pos, v in agg.items() if pos]
    fixed_rate = (sum(c for c, _ in fixed_all) / sum(e for _, e in fixed_all)) if fixed_all and sum(e for _, e in fixed_all) else None
    out = []
    for (a, b), x in cards.items():
        if x["exposures"] < MIN_EXPOSURES:
            continue
        pos = position.get((a, b), 0)
        p0 = prior.get(pos) or 0.0
        if not p0:
            continue
        rate = (x["clicks"] + PRIOR_EXPOSURES * p0) / (x["exposures"] + PRIOR_EXPOSURES)
        factor = max(FLOOR, min(CAP, rate / p0))
        out.append({"a": a, "b": b, "shown_as": "fixed" if pos else "rotation", "position": pos,
                    "exposures": round(x["exposures"], 1), "clicks": int(x["clicks"]), "sales": int(x["sales"]),
                    "revenue": x["revenue"], "rate": round(rate, 5), "factor": round(factor, 3)})
    fb = {"shop": shop_key, "window": f"{start}..{end}", "made": str(dt.date.today()),
          "prior_by_position": prior, "cards": out}
    path = ROOT / "cache" / shop_key / "feedback.json"
    path.write_text(json.dumps(fb, indent=1))
    fixed = [c for c in out if c["shown_as"] == "fixed"]
    summary = {
        "window": fb["window"], "live_sets": st["live_sets"], "cards_scored": len(out),
        "fixed_cards_scored": len(fixed), "fixed_click_rate": fixed_rate,
        "rotation_click_rate": prior.get(0),
        "fixed_winners": sum(c["factor"] >= 1.2 for c in fixed),
        "fixed_losers": sum(c["factor"] <= 0.8 for c in fixed),
        "fixed_sales": sum(c["sales"] for c in fixed),
        "rotation_pairs_scored": len(out) - len(fixed),
        "rotation_pairs_above_avg": sum(c["factor"] >= 1.2 for c in out if c["shown_as"] == "rotation"),
    }
    if build:
        summary["next_run"] = _rebuild(shop_key)
    (ROOT / "out" / shop_key / f"learn_{end}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    return summary


def _rebuild(shop_key: str) -> dict:
    """Next sets with fresh inputs and the feedback, and how many cards change vs what is live."""
    import subprocess
    import sys
    subprocess.run([sys.executable, str(ROOT / "run.py"), "build", "--shop", shop_key, "--refresh"],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    run_dir = ROOT / "out" / shop_key / dt.date.today().isoformat()
    s = json.loads((run_dir / "summary.json").read_text())
    return {"run": str(run_dir.relative_to(ROOT)), "vs_live": s.get("vs_live"), "feedback": s.get("feedback")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="demo", choices=sorted(SHOPS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", help="default: yesterday")
    ap.add_argument("--no-build", action="store_true", help="score the cards only, do not rebuild")
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=1)
    learn(a.shop, dt.date.fromisoformat(a.start), end, build=not a.no_build)


if __name__ == "__main__":
    main()
