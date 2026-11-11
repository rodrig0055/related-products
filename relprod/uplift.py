"""Estimate what better related products are worth, before any test, from the rotation itself.

The rotation shows every product in a PDP's pool with equal probability (4 random cards from the
products sharing a category or subcategory, the storefront default), so it is a natural
experiment: a pair (A, B) was shown about views(A) x 4 / |pool(A)| times, and GA4 shows how often
it was clicked and the clicked product bought.

Out of sample: the engine picks its sets from behaviour up to the day before the evaluation
window only, and is scored on clicks and sales inside the window. Click-through and sale value are
modelled per relevance band (relevance as the engine computed it from the earlier data), then:

  rotation  = sum over A, B in pool(A) of exposure(A, B) x rate(band(A, B))   (sanity: ~observed)
  engine    = sum over A, B in set(A)  of views(A)       x rate(band(A, B))

  uv run python -m relprod.uplift --shop demo --days 90
"""

import argparse
import datetime as dt
import json
from collections import defaultdict

from .engine import Engine
from .features import build_products
from .measure import click_purchases, flow
from .shops import ROOT, SHOPS
from . import sources

BANDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 1.01]


def band(x: float) -> int:
    return next(i for i in range(len(BANDS) - 1) if BANDS[i] <= x < BANDS[i + 1])


def rotation_pools(cat: dict, products: dict) -> dict[int, set[int]]:
    """Storefront rotation pool: products sharing any category or subcategory (the storefront
    also hides unpublished and search-excluded products, which `products` already leaves out)."""
    cats, subs = defaultdict(set), defaultdict(set)
    for r in cat["subs"]:
        pid = int(r["product_id"])
        if pid in products:
            cats[pid].add(r["cat"])
            subs[pid].add(r["sub"])
    by_cat, by_sub = defaultdict(set), defaultdict(set)
    for pid in products:
        for c in cats[pid]:
            by_cat[c].add(pid)
        for s in subs[pid]:
            by_sub[s].add(pid)
    eligible = {pid for pid, p in products.items() if not p.not_linkable_why.startswith("excluded")}
    return {a: (set().union(*(by_cat[c] for c in cats[a]), *(by_sub[s] for s in subs[a])) & eligible) - {a}
            for a in products}


def estimate(shop_key: str, days: int) -> dict:
    shop = SHOPS[shop_key]
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days - 1)
    train_end = start - dt.timedelta(days=1)

    cat = sources.catalog(shop)
    beh = sources.behaviour(shop, days=365, end=train_end)
    products = build_products(shop, cat, sources.sales(shop), beh, sources.search_demand(shop), sources.link_equity(shop),
                              sources.hub_sales(shop))
    eng = Engine(products, beh)
    picks, _ = eng.run()
    sets = {a: [pk.target for pk in c] for a, c in picks.items()}
    pools = rotation_pools(cat, products)

    fl = flow(shop.fronts[0], start, end)   # the first storefront
    slug_id = {p.slug: p.id for p in products.values()}
    views = {slug_id[s]: n for s, n in fl["views"].items() if s in slug_id}
    clicks = click_purchases(fl, slug_id)

    # Exposure, clicks, sales and revenue per relevance band, over rotation pairs.
    expo = defaultdict(float)
    hit = defaultdict(lambda: {"clicks": 0, "sales": 0, "revenue": 0.0})
    for a, pool in pools.items():
        if not views.get(a) or not pool:
            continue
        e = views[a] * min(1.0, 4 / len(pool))
        for b in pool:
            expo[band(eng.relevance(products[a], products[b])[0] if eng.allowed(products[a], products[b]) else 0.0)] += e
    in_pool = 0
    for c in clicks:
        a, b = slug_id.get(c["src"]), slug_id.get(c["dst"])
        if a is None or b is None or b not in pools.get(a, ()):
            continue
        in_pool += 1
        k = band(eng.relevance(products[a], products[b])[0] if eng.allowed(products[a], products[b]) else 0.0)
        hit[k]["clicks"] += 1
        hit[k]["sales"] += c["bought"]
        hit[k]["revenue"] += c["revenue"]
    rate = {k: {"ctr": hit[k]["clicks"] / expo[k] if expo[k] else 0.0,
                "rev_per_impr": hit[k]["revenue"] / expo[k] if expo[k] else 0.0,
                "sales_per_impr": hit[k]["sales"] / expo[k] if expo[k] else 0.0} for k in range(len(BANDS) - 1)}

    # Sales after a click are sparse (tens per quarter), so value per click is pooled into two
    # levels (relevance below / from 0.20) instead of per band; clicks stay per band.
    split = BANDS.index(0.20)
    lvl = lambda k: "high" if k >= split else "low"
    pooled = defaultdict(lambda: {"clicks": 0, "sales": 0, "revenue": 0.0})
    for k, h in hit.items():
        for f in ("clicks", "sales", "revenue"):
            pooled[lvl(k)][f] += h[f]
    for k in rate:
        g = pooled[lvl(k)]
        rate[k]["rev_per_impr"] = rate[k]["ctr"] * (g["revenue"] / g["clicks"] if g["clicks"] else 0)
        rate[k]["sales_per_impr"] = rate[k]["ctr"] * (g["sales"] / g["clicks"] if g["clicks"] else 0)

    def predict(pairs_by_a, per_a_weight):
        out = {"clicks": 0.0, "sales": 0.0, "revenue": 0.0}
        for a, bs in pairs_by_a.items():
            if not views.get(a) or not bs:
                continue
            w = per_a_weight(a, bs)
            for b in bs:
                k = band(eng.relevance(products[a], products[b])[0] if eng.allowed(products[a], products[b]) else 0.0)
                out["clicks"] += views[a] * w * rate[k]["ctr"]
                out["sales"] += views[a] * w * rate[k]["sales_per_impr"]
                out["revenue"] += views[a] * w * rate[k]["rev_per_impr"]
        return out

    rot = predict(pools, lambda a, bs: min(1.0, 4 / len(bs)))
    new = predict(sets, lambda a, bs: 1.0)
    observed = {"clicks": in_pool, "sales": sum(h["sales"] for h in hit.values()),
                "revenue": sum(h["revenue"] for h in hit.values())}
    month = 30 / days
    sess = fl["sessions"]
    return {
        "window": f"{start}..{end}", "train_end": str(train_end), "days": days,
        "pdp_views": sum(views.values()), "related_clicks_all": len(clicks), "related_clicks_in_pool": in_pool,
        "ga4_revenue": sum(s["revenue"] for s in sess), "ga4_buying_sessions": sum(s["bought"] for s in sess),
        "bands": [{"relevance": f"{BANDS[k]:.2f}-{min(BANDS[k + 1], 1):.2f}", "impressions": round(expo[k]),
                   **hit[k], "ctr_pct": round(rate[k]["ctr"] * 100, 2),
                   "rev_per_1000_impr": round(rate[k]["rev_per_impr"] * 1000, 1)} for k in range(len(BANDS) - 1)],
        "value_per_click": {k: {"clicks": g["clicks"], "sales": g["sales"], "revenue": g["revenue"],
                                "revenue_per_click": round(g["revenue"] / g["clicks"], 2) if g["clicks"] else 0}
                            for k, g in pooled.items()},
        "observed": observed, "rotation_model": rot, "engine_model": new,
        "per_month": {"rotation": {k: v * month for k, v in rot.items()},
                      "engine": {k: v * month for k, v in new.items()},
                      "delta": {k: (new[k] - rot[k]) * month for k in rot}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="demo", choices=sorted(SHOPS))
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()
    res = estimate(args.shop, args.days)
    out = ROOT / "out" / args.shop / f"uplift_{dt.date.today()}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
