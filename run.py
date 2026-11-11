"""Related-products engine.

  uv run run.py build --shop demo            # uses cached inputs (cache/<shop>/) where present
  uv run run.py build --shop demo --refresh    # re-pull catalog, sales, GA4, GSC

Writes out/<shop>/<date>/: product_related.csv + product_related.sql (the CMS table's shape),
related.csv (one row per card, with the reason), cms.csv (one row per product: the 4 related ids in
display order), products.csv, summary.json, data.json (report).
"""

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
from collections import Counter, defaultdict

from relprod import config as C
from relprod import seasons as S
from relprod import sources
from relprod.engine import Engine
from relprod.features import build_products
from relprod.shops import ROOT, SHOPS


def rotation_inbound(cat: dict, products: dict) -> dict[int, float]:
    """Expected related links per product under the old rotation (4 random cards from the products
    sharing any published subcategory), for the before/after comparison."""
    members, of = defaultdict(set), defaultdict(set)
    for r in cat["subs"]:
        pid = int(r["product_id"])
        if pid in products and int(r["sub_published"] or 0):
            members[r["sub"]].add(pid)
            of[pid].add(r["sub"])
    exp: Counter = Counter()
    for a, subs in of.items():
        pool = set().union(*(members[s] for s in subs)) - {a}
        pool = {b for b in pool if products[b].linkable}
        for b in pool:
            exp[b] += min(1.0, C.CARDS / len(pool))
    return exp


def build(args) -> None:
    shop = SHOPS[args.shop]
    cat = sources.catalog(shop, args.refresh)
    sal = sources.sales(shop, args.refresh)
    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    active = S.active(shop.key, day)
    beh = sources.behaviour(shop, args.refresh)
    for season in active:
        if season.last_year and beh:
            lo, hi = season.last_year
            last = sources.behaviour(shop, False, days=(hi - lo).days + 1, end=hi, pageviews=True)
            for k in ("coview", "subst"):
                for pair, n in last.get(k, {}).items():
                    beh[k][pair] = beh[k].get(pair, 0) + S.LAST_YEAR_WEIGHT * n
    dem = sources.search_demand(shop, args.refresh)
    eq = sources.link_equity(shop)
    hs = sources.hub_sales(shop, args.refresh)
    colls = sources.collections(shop, args.refresh)
    products = build_products(shop, cat, sal, beh, dem, eq, hs, colls)
    print(f"{shop.name}: {len(products)} products, {sum(p.linkable for p in products.values())} linkable", file=sys.stderr)

    fb_path = ROOT / "cache" / shop.key / "feedback.json"
    feedback = json.loads(fb_path.read_text()) if fb_path.exists() and not args.no_feedback else None
    tags = frozenset().union(*(s.tags for s in active)) if active else frozenset()
    eng = Engine(products, beh, feedback, tags, S.SEASON_BOOST)
    if active:
        print(f"seasons on {day}: {', '.join(s.name for s in active)}", file=sys.stderr)
    pinned = sources.current_related(shop) if args.keep_live else None
    if pinned:
        print(f"keeping {len(pinned)} live sets as they are: {sorted(pinned)}", file=sys.stderr)
    picks, inbound = eng.run(pinned=pinned)
    kit = {a: c for a, c in picks.items() if a not in (pinned or {})}  # files hold only this run's sets
    before = rotation_inbound(cat, products)

    out = ROOT / "out" / shop.key / (dt.date.today().isoformat() + (f"_for_{day}" if args.date else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for aid, chosen in picks.items():
        if aid in (pinned or {}):
            continue  # kept live as it is: not part of this kit's files
        a = products[aid]
        for pk in sorted(chosen, key=lambda x: x.position):
            b = products[pk.target]
            rows.append({"source_id": a.id, "source_slug": a.slug, "source_name": a.name, "position": pk.position,
                         "slot": pk.slot, "related_id": b.id, "related_slug": b.slug, "related_name": b.name,
                         "relevance": round(pk.relevance, 3), "content_similarity": round(pk.tags, 3),
                         **{f"sim_{k}": round(v, 3) for k, v in eng.content(a, b).items() if k != "content"},
                         "coview_sessions": pk.coview, "bought_instead_sessions": pk.subst,
                         "price_ratio": round(b.price / a.price, 2) if a.price else "",
                         "band_widened": pk.band_step, "coverage_fix": pk.repaired, "reason": pk.reason})
    with (out / "related.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "cms.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "product_slug"] + [f"related_{i}_id" for i in range(1, C.CARDS + 1)])
        for aid, chosen in sorted(kit.items()):
            ids = [pk.target for pk in sorted(chosen, key=lambda x: x.position)]
            w.writerow([aid, products[aid].slug] + ids + [""] * (C.CARDS - len(ids)))
    prod_rows = []
    for p in sorted(products.values(), key=lambda x: -x.views):
        prod_rows.append({"id": p.id, "slug": p.slug, "name": p.name, "price": p.price, "price_source": p.price_source,
                          "views_12m": p.views, "bought_12m": p.bought, "cvr_smoothed": round(p.cvr, 4),
                          "orders_12m": p.orders, "nonbrand_impressions": p.search_impressions,
                          "true_pr": p.true_pr, "linkable": p.linkable, "not_linkable_why": p.not_linkable_why,
                          "inbound_after": inbound[p.id], "inbound_before_expected": round(before.get(p.id, 0), 2),
                          "occasions": " ".join(sorted(p.facets.get("occasion", []))),
                          "recipients": " ".join(sorted(p.facets.get("recipient", []))),
                          "kind": " ".join(sorted(p.facets.get("kind", []))),
                          "flowers": " ".join(sorted(p.facets.get("flower", []))),
                          "colours": " ".join(sorted(p.colours))})
    with (out / "products.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prod_rows[0]))
        w.writeheader()
        w.writerows(prod_rows)

    # The CMS table: product_related(product_id, related_product_id, prioritization). Lower
    # prioritization shows first (the convention of every *_prioritization column in this DB).
    ordered = {aid: [pk.target for pk in sorted(c, key=lambda x: x.position)] for aid, c in sorted(kit.items())}
    with (out / "product_related.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "related_product_id", "prioritization"])
        for aid, ids in ordered.items():
            for i, rid in enumerate(ids, 1):
                w.writerow([aid, rid, i])
    values = ",\n".join(f"({aid},{rid},{i})" for aid, ids in ordered.items() for i, rid in enumerate(ids, 1))
    (out / "product_related.sql").write_text(
        f"-- {shop.name} related products, generated {dt.date.today()} by related-products.\n"
        f"-- Replaces the sets of the {len(ordered)} products below; other products are untouched.\n"
        f"START TRANSACTION;\n"
        f"DELETE FROM product_related WHERE product_id IN ({','.join(str(a) for a in ordered)});\n"
        f"INSERT INTO product_related (product_id, related_product_id, prioritization) VALUES\n{values};\n"
        f"COMMIT;\n")
    live = sources.current_related(shop)
    diff = None
    if live is not None:
        diff = {"live_products": len(live),
                "unchanged": sum(1 for a, ids in ordered.items() if live.get(a) == ids),
                "changed": sum(1 for a, ids in ordered.items() if a in live and live[a] != ids),
                "new": sum(1 for a in ordered if a not in live),
                "live_only": sorted(a for a in live if a not in ordered)}

    linkable = [p for p in products.values() if p.linkable]
    filled = Counter(len(c) for c in picks.values())
    summary = {
        "shop": shop.name, "date": dt.date.today().isoformat(),
        "inputs": {"catalog": len(products), "sales_products": len(sal),
                   "ga4": f"{beh['start']}..{beh['end']}" if beh else None,
                   "ga4_coview_pairs": len(beh.get("coview", {})) // 2 if beh else 0,
                   "ga4_bought_instead_pairs": len(beh.get("subst", {})) if beh else 0,
                   "gsc_products": len(dem), "tipr_products": len(eq)},
        "pdps": len(picks), "linkable": len(linkable),
        "not_linkable": Counter(p.not_linkable_why.split(" (")[0] for p in products.values() if not p.linkable),
        "cards_filled": {str(k): v for k, v in sorted(filled.items())},
        "slots": {s["id"]: Counter() for s in C.SLOTS},
        "with_behaviour_evidence": sum(1 for r in rows if r["coview_sessions"] or r["bought_instead_sessions"]) / len(rows),
        "coverage_fixes": sum(1 for r in rows if r["coverage_fix"]),
        "inbound_after": {"zero": sum(1 for p in linkable if inbound[p.id] == 0),
                          "below_min": sum(1 for p in linkable if inbound[p.id] < C.MIN_INBOUND),
                          "max": max(inbound[p.id] for p in linkable),
                          "median": statistics.median(inbound[p.id] for p in linkable)},
        "inbound_before_expected": {"below_min": sum(1 for p in linkable if before.get(p.id, 0) < C.MIN_INBOUND),
                                    "max": round(max(before.get(p.id, 0) for p in linkable), 1),
                                    "median": round(statistics.median(before.get(p.id, 0) for p in linkable), 2)},
        "vs_live": diff,
        # Products in a strict lane that also carry other occasions: likely mis-tagged, for review.
        "strict_lane_mixed_tags": [{"id": p.id, "name": p.name,
                                    "occasions": sorted(p.facets.get("occasion", set()))}
                                   for p in products.values()
                                   if p.strict and p.facets.get("occasion", set()) - p.strict],
        "seasons": [{"key": s.key, "name": s.name, "tags": sorted(s.tags),
                     "last_year": [str(d) for d in s.last_year] if s.last_year else None} for s in active],
        "feedback": ({"window": feedback["window"], "cards": len(eng.feedback),
                      "cards_used": sum(1 for r in rows if (r["source_id"], r["related_id"]) in eng.feedback)}
                     if feedback else None),
        "config": {"min_inbound": C.MIN_INBOUND, "max_inbound": C.MAX_INBOUND, "min_relevance": C.MIN_RELEVANCE},
    }
    for r in rows:
        summary["slots"][r["slot"]]["band_" + str(r["band_widened"])] += 1
    summary["kept_live"] = sorted(pinned or {})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=dict))

    data = {"summary": summary, "slots": C.SLOTS,
            "products": {p.id: {"slug": p.slug, "name": p.name, "url": p.url, "image": p.image, "price": p.price,
                                "views": p.views, "orders": p.orders, "cvr": round(p.cvr, 4),
                                "impr": p.search_impressions, "linkable": p.linkable,
                                "why": p.not_linkable_why, "in": inbound[p.id], "before": round(before.get(p.id, 0), 1),
                                "occ": sorted(p.facets.get("occasion", [])), "rec": sorted(p.facets.get("recipient", [])),
                                "kind": sorted(p.facets.get("kind", [])), "flw": sorted(p.facets.get("flower", []))}
                         for p in products.values()},
            "picks": {aid: [{"pos": pk.position, "slot": pk.slot, "id": pk.target, "rel": round(pk.relevance, 2),
                             "reason": pk.reason, "fix": pk.repaired} for pk in sorted(c, key=lambda x: x.position)]
                      for aid, c in picks.items()}}
    (out / "data.json").write_text(json.dumps(data, default=list))
    print(json.dumps(summary, indent=1, default=dict))
    print(f"wrote {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--shop", default="demo", choices=sorted(SHOPS))
    b.add_argument("--refresh", action="store_true", help="re-pull GA4 and GSC from BigQuery instead of cache/<shop>/")
    b.add_argument("--date", help="build for this day's seasons (seasons.py); output dir gets _for_<date>")
    b.add_argument("--keep-live", action="store_true",
                   help="leave products that already have rows in product_related exactly as they are")
    b.add_argument("--no-feedback", action="store_true", help="ignore cache/<shop>/feedback.json (learn.py)")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
