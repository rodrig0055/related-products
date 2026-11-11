"""Engine sets vs the rotation, split by product page in traffic-matched pairs. No tracking deploy:
everything is read from the GA4 export (see measure.py for how a related click is detected).

  uv run python -m relprod.abtest assign --run out/demo/<date>       # pairs, arms, treatment sets
  uv run python -m relprod.abtest aa                                 # same split on past data: expect ~0
  uv run python -m relprod.abtest results --start 2026-10-20         # live results, any day after start

  uv run python -m relprod.abtest assign --shop demo --run out/demo/<date> --top 400
  uv run python -m relprod.abtest results --shop demo --start 2026-11-03

Shops whose fronts share one related-products table: a product is in the same arm on every
front. Pairs are matched on both fronts' views together; each front is read on its own (--front).
--top limits the test to the busiest products (on one 4,000-product shop, 400 carry ~80% of PDP views); the
rest keep the rotation until the test ends.

Arms: "engine" products get their 4 cards in product_related; "rotation" products get no rows,
so the storefront keeps drawing 4 random cards for them (the storefront's default rotation).
Each lift compares the two products within every pair (weighted by the pair's traffic, capped at
the 90th percentile so Rosy Glow cannot decide it), adjusted with the same pair's difference in
the 6 weeks before (CUPED), as a share of the rotation rate. The 95% interval comes from
re-flipping the coins inside the pairs (randomization inference; validate.py showed resampling
pairs was too narrow).
"""

import argparse
import csv
import datetime as dt
import json
import random
from collections import defaultdict

from .measure import click_purchases, flow
from .shops import ROOT, SHOPS
from . import sources

SEED = 20261019
PRE_DAYS = 42
BOOT = 2000
N_NULL = 400   # coin re-flips for the chance band


def kit_dir(shop_key: str):
    d = ROOT / "out" / shop_key / "abtest"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- assignment ------------------------------------------------------------------------------

def front_of(shop, key: str | None):
    fronts = {f.key: f for f in shop.fronts}
    return fronts[key] if key else shop.fronts[0]


def assign(shop_key: str, run: str, top: int | None = None, skip: int = 0, exclude: frozenset = frozenset()) -> None:
    shop = SHOPS[shop_key]
    run_dir = ROOT / run
    cat = sources.catalog(shop)
    names = {int(p["id"]): (p["slug"], p["name"]) for p in cat["products"]}
    sets: dict[int, list[int]] = defaultdict(list)
    with open(run_dir / "related.csv") as f:
        for r in sorted(csv.DictReader(f), key=lambda r: (int(r["source_id"]), int(r["position"]))):
            sets[int(r["source_id"])].append(int(r["related_id"]))
    sets = {k: v for k, v in sets.items() if len(v) == 4 and k not in exclude}
    end = dt.date.today() - dt.timedelta(days=1)
    views: dict[int, int] = defaultdict(int)
    for fr in shop.fronts:
        if fr.ga4_table:
            fl = flow(fr, end - dt.timedelta(days=89), end)
            for pid, (slug, _) in names.items():
                views[pid] += fl["views"].get(slug, 0)
    order = sorted(sets, key=lambda pid: (-views.get(pid, 0), pid))[skip:]
    if top:
        order = order[:top - top % 2]
    rng = random.Random(SEED)
    rows = []
    for i in range(0, len(order), 2):
        pair = order[i:i + 2]
        arms = ["engine", "rotation"] if rng.random() < 0.5 else ["rotation", "engine"]
        if len(pair) == 1:
            arms = ["rotation"]  # odd one out stays on rotation and is left out of the analysis
        for pid, arm in zip(pair, arms):
            rows.append({"product_id": pid, "slug": names[pid][0], "name": names[pid][1],
                         "views_90d": views.get(pid, 0), "pair": i // 2 if len(pair) == 2 else -1, "arm": arm})
    d = kit_dir(shop_key)
    with open(d / "assignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    treat = {r["product_id"]: sets[r["product_id"]] for r in rows if r["arm"] == "engine"}
    (d / "treatment_sets.json").write_text(json.dumps(
        {"source_run": run, "assigned": str(dt.date.today()), "sets": {str(k): v for k, v in treat.items()}}, indent=1))
    values = ",\n".join(f"({a},{b},{i})" for a, ids in sorted(treat.items()) for i, b in enumerate(ids, 1))
    (d / "product_related_treatment.sql").write_text(
        f"-- {shop.name} related-products test, engine arm only ({len(treat)} products). Rotation arm: no rows.\n"
        f"START TRANSACTION;\nDELETE FROM product_related WHERE product_id IN ({','.join(map(str, sorted(treat)))});\n"
        f"INSERT INTO product_related (product_id, related_product_id, prioritization) VALUES\n{values};\nCOMMIT;\n")
    with open(d / "product_related_treatment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "related_product_id", "prioritization"])
        for a, ids in sorted(treat.items()):
            w.writerows([a, b, i] for i, b in enumerate(ids, 1))
    # Where the admin screen for related products lives (admin_origin, else the storefront).
    admin = shop.admin_origin or shop.origin
    with open(d / "upload_checklist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "product", "admin_url"] + [f"related_{i}" for i in range(1, 5)])
        for r in rows:
            if r["arm"] == "engine":
                pid = r["product_id"]
                w.writerow([pid, names[pid][1], f"{admin}/admin/products/manage-related-products/{pid}"]
                           + [f"{names[b][1]} ({b})" for b in treat[pid]])
    t = [r for r in rows if r["arm"] == "engine"]
    c = [r for r in rows if r["arm"] == "rotation" and r["pair"] >= 0]
    print(f"{len(t)} engine / {len(c)} rotation products in pairs; views 90d: "
          f"engine {sum(r['views_90d'] for r in t):,}, rotation {sum(r['views_90d'] for r in c):,}")
    print(f"wrote {d}")


def refresh(shop_key: str, run: str) -> None:
    """Mid-test refresh (a season switching on or off, or learn.py's next sets): the engine arm
    takes the new run's sets, arms and pairs stay as they are, the rotation arm stays without rows.
    The old plan is kept in treatment_history/ and the watcher checks against the new one."""
    shop = SHOPS[shop_key]
    d = kit_dir(shop_key)
    rows = load_assignment(shop_key)
    cat = sources.catalog(shop)
    names = {int(p["id"]): p["name"] for p in cat["products"]}
    new: dict[int, list[int]] = defaultdict(list)
    with open(ROOT / run / "related.csv") as f:
        for r in sorted(csv.DictReader(f), key=lambda r: (int(r["source_id"]), int(r["position"]))):
            new[int(r["source_id"])].append(int(r["related_id"]))
    old = json.loads((d / "treatment_sets.json").read_text())
    hist = d / "treatment_history"
    hist.mkdir(exist_ok=True)
    (hist / f"treatment_sets_until_{dt.date.today()}.json").write_text(json.dumps(old, indent=1))
    treat = {r["product_id"]: (new.get(r["product_id"]) if len(new.get(r["product_id"], [])) == 4
                               else old["sets"][str(r["product_id"])])
             for r in rows if r["arm"] == "engine"}
    changed = {a: ids for a, ids in treat.items() if ids != old["sets"][str(a)]}
    (d / "treatment_sets.json").write_text(json.dumps(
        {"source_run": run, "assigned": old["assigned"], "refreshed": str(dt.date.today()),
         "sets": {str(k): v for k, v in treat.items()}}, indent=1))
    values = ",\n".join(f"({a},{b},{i})" for a, ids in sorted(changed.items()) for i, b in enumerate(ids, 1))
    (d / f"refresh_{dt.date.today()}.sql").write_text(
        f"-- {shop.name} test refresh {dt.date.today()} from {run}: engine-arm sets that change ({len(changed)}).\n"
        f"START TRANSACTION;\nDELETE FROM product_related WHERE product_id IN ({','.join(map(str, sorted(changed))) or 'NULL'});\n"
        + (f"INSERT INTO product_related (product_id, related_product_id, prioritization) VALUES\n{values};\n" if changed else "")
        + "COMMIT;\n")
    admin = shop.admin_origin or shop.origin
    with open(d / f"refresh_{dt.date.today()}_checklist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "product", "admin_url"] + [f"related_{i}" for i in range(1, 5)])
        for a, ids in sorted(changed.items(), key=lambda kv: -next(int(r["views_90d"]) for r in rows if r["product_id"] == kv[0])):
            w.writerow([a, names.get(a, a), f"{admin}/admin/products/manage-related-products/{a}"]
                       + [f"{names.get(b, b)} ({b})" for b in ids])
    # The full plan files follow the latest plan too, so an upload not done yet uses current sets.
    full = ",\n".join(f"({a},{b},{i})" for a, ids in sorted(treat.items()) for i, b in enumerate(ids, 1))
    (d / "product_related_treatment.sql").write_text(
        f"-- {shop.name} related-products test, engine arm only ({len(treat)} products), plan of {dt.date.today()}. "
        f"Rotation arm: no rows.\nSTART TRANSACTION;\nDELETE FROM product_related WHERE product_id IN "
        f"({','.join(map(str, sorted(treat)))});\n"
        f"INSERT INTO product_related (product_id, related_product_id, prioritization) VALUES\n{full};\nCOMMIT;\n")
    with open(d / "product_related_treatment.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "related_product_id", "prioritization"])
        for a, ids in sorted(treat.items()):
            w.writerows([a, b, i] for i, b in enumerate(ids, 1))
    with open(d / "upload_checklist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "product", "admin_url"] + [f"related_{i}" for i in range(1, 5)])
        for r in rows:
            if r["arm"] == "engine":
                a = r["product_id"]
                w.writerow([a, names.get(a, a), f"{admin}/admin/products/manage-related-products/{a}"]
                           + [f"{names.get(b, b)} ({b})" for b in treat[a]])
    print(f"{len(changed)} of {len(treat)} engine-arm sets change; wrote refresh_{dt.date.today()}.sql/_checklist.csv "
          f"and updated the full plan files")


def load_assignment(shop_key: str) -> list[dict]:
    with open(kit_dir(shop_key) / "assignment.csv") as f:
        return [{**r, "product_id": int(r["product_id"]), "pair": int(r["pair"])} for r in csv.DictReader(f)]


# --- measurement ---------------------------------------------------------------------------------

def per_product(front, start: dt.date, end: dt.date, slug_id: dict[str, int], segment: bool = True) -> dict[int, dict]:
    """Per source product: PDP views, related clicks from it, direct sales of clicked products, and
    the sessions that landed on it (first PDP): more than one PDP, bought, revenue."""
    fl = flow(front, start, end, segment=segment)
    m: dict[int, dict] = defaultdict(lambda: dict.fromkeys(
        ("views", "clicks", "click_sales", "click_revenue", "landings", "multi", "buyers", "revenue"), 0))
    for slug, n in fl["views"].items():
        if slug in slug_id:
            m[slug_id[slug]]["views"] += n
    for c in click_purchases(fl, slug_id):
        if c["src"] in slug_id:
            x = m[slug_id[c["src"]]]
            x["clicks"] += 1
            x["click_sales"] += c["bought"]
            x["click_revenue"] += c["revenue"]
    for s in fl["sessions"]:
        if s["first"] in slug_id:
            x = m[slug_id[s["first"]]]
            x["landings"] += 1
            x["multi"] += s["pdps"] > 1
            x["buyers"] += s["bought"]
            x["revenue"] += s["revenue"]
    return m


METRICS = {   # name: (numerator, denominator, label)
    "click_rate": ("clicks", "views", "Related clicks per PDP view"),
    "multi_pdp": ("multi", "landings", "Landing sessions viewing 2+ PDPs"),
    "click_sales": ("click_revenue", "views", "Direct sales of clicked products per PDP view"),
    "conversion": ("buyers", "landings", "Landing sessions that bought"),
    "revenue": ("revenue", "landings", "Revenue per landing session"),
}


def _pair_rows(pairs, data, arm_of, metric):
    """Per pair: (engine rate - rotation rate, rotation rate, weight). Weight = the pair's smaller
    denominator, capped, so one giant product (Rosy Glow is ~13% of PDP views) cannot decide
    the result and a pair with no traffic on one side counts for nothing."""
    num, den = METRICS[metric][:2]
    out = {}
    for i, members in enumerate(pairs):
        by_arm = {arm_of[pid]: data.get(pid) for pid in members}
        e, r = by_arm.get("engine"), by_arm.get("rotation")
        if not e or not r or not e[den] or not r[den]:
            continue
        out[i] = (e[num] / e[den] - r[num] / r[den], r[num] / r[den], min(e[den], r[den]))
    return out


def _estimate(idx, test_rows, pre_rows, cap):
    """Weighted mean pair difference, CUPED-adjusted with the same pair's pre-period difference,
    as a share of the weighted rotation rate."""
    both = [i for i in idx if i in test_rows]
    if not both:
        return None
    w = [min(test_rows[i][2], cap) for i in both]
    y = [test_rows[i][0] for i in both]
    x = [pre_rows[i][0] if i in pre_rows else 0.0 for i in both]
    sw = sum(w)
    mx, my = sum(wi * xi for wi, xi in zip(w, x)) / sw, sum(wi * yi for wi, yi in zip(w, y)) / sw
    vx = sum(wi * (xi - mx) ** 2 for wi, xi in zip(w, x))
    theta = sum(wi * (xi - mx) * (yi - my) for wi, xi, yi in zip(w, x, y)) / vx if vx else 0.0
    adj = sum(wi * (yi - theta * (xi - mx)) for wi, xi, yi in zip(w, x, y)) / sw
    base = sum(wi * test_rows[i][1] for wi, i in zip(w, both)) / sw
    return adj / base if base else None


def compare(shop_key: str, start: dt.date, end: dt.date, front: str | None = None, segment: bool = True) -> dict:
    shop = SHOPS[shop_key]
    fr = front_of(shop, front)
    rows = load_assignment(shop_key)
    slug_id = {r["slug"]: r["product_id"] for r in rows}
    arm_of = {r["product_id"]: r["arm"] for r in rows}
    pairs = defaultdict(list)
    for r in rows:
        if r["pair"] >= 0:
            pairs[r["pair"]].append(r["product_id"])
    pairs = list(pairs.values())
    pre_end = start - dt.timedelta(days=1)
    pre = per_product(fr, pre_end - dt.timedelta(days=PRE_DAYS - 1), pre_end, slug_id, segment)
    test = per_product(fr, start, end, slug_id, segment)
    rng = random.Random(SEED)
    out = {"front": fr.key, "front_name": fr.name, "window": f"{start}..{end}", "pre_window": f"{pre_end - dt.timedelta(days=PRE_DAYS - 1)}..{pre_end}",
           "pairs": len(pairs), "segment": fr.ga4_segment if segment else None, "metrics": {}}
    # Chance band by randomization inference: re-flip the coin inside every pair N_NULL times on
    # the same data and re-estimate. The pair bootstrap was too narrow on past data (validate.py),
    # so the 95% interval is the point estimate minus that band (a result is clear when the
    # estimate lies outside what re-flipped coins produce).
    pair_ids = [tuple(m) for m in pairs]
    flips = []
    for i in range(N_NULL):
        r_ = random.Random(SEED + i)
        f = {}
        for a, b in pair_ids:
            x, y = ("engine", "rotation") if r_.random() < 0.5 else ("rotation", "engine")
            f[a], f[b] = x, y
        flips.append(f)
    for metric, (num, den, label) in METRICS.items():
        tr, pr = _pair_rows(pairs, test, arm_of, metric), _pair_rows(pairs, pre, arm_of, metric)
        ws = sorted(v[2] for v in tr.values())
        cap = ws[int(0.9 * (len(ws) - 1))] if ws else 1   # no pair weighs more than the 90th percentile
        idx = list(tr)
        point = _estimate(idx, tr, pr, cap)
        null = []
        for f in flips:
            t2, p2 = _pair_rows(pairs, test, f, metric), _pair_rows(pairs, pre, f, metric)
            if t2:
                w2 = sorted(v[2] for v in t2.values())
                v = _estimate(list(t2), t2, p2, w2[int(0.9 * (len(w2) - 1))])
                if v is not None:
                    null.append(v)
        null.sort()
        band = (null[int(0.025 * len(null))], null[int(0.975 * len(null)) - 1]) if null else None
        tot = {a: [sum(test.get(pid, {}).get(num, 0) for m in pairs for pid in m if arm_of[pid] == a),
                   sum(test.get(pid, {}).get(den, 0) for m in pairs for pid in m if arm_of[pid] == a)]
               for a in ("engine", "rotation")}
        out["metrics"][metric] = {
            "label": label, "pairs_used": len(idx),
            "test": {a: (n / d if d else 0.0) for a, (n, d) in tot.items()}, "counts_test": tot,
            "lift": point, "chance_band95": band,
            "ci95": [point - band[1], point - band[0]] if point is not None and band else None,
            "p_value": (sum(abs(v) >= abs(point) for v in null) / len(null)) if point is not None and null else None,
        }
    return out


def show(res: dict) -> None:
    print(f"{res.get('front_name', '')} window {res['window']} vs pre {res['pre_window']}, {res['pairs']} pairs")
    for k, m in res["metrics"].items():
        lift = "n/a" if m["lift"] is None else f"{m['lift'] * 100:+.1f}%"
        ci = "" if not m["ci95"] else f"  95% CI {m['ci95'][0] * 100:+.1f}% to {m['ci95'][1] * 100:+.1f}%"
        print(f"  {m['label']:48} engine {m['test']['engine']:.4f}  rotation {m['test']['rotation']:.4f}  "
              f"lift {lift}{ci}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assign")
    a.add_argument("--run", required=True, help="engine run dir, e.g. out/demo/<date>")
    a.add_argument("--top", type=int, help="test only the N busiest products (both fronts' views)")
    a.add_argument("--skip", type=int, default=0, help="leave the N busiest out of the test (they get engine sets)")
    a.add_argument("--exclude", default="", help="comma-separated product ids kept out of the test (e.g. hand-entered sets)")
    rf = sub.add_parser("refresh")
    rf.add_argument("--run", required=True, help="new engine run dir for the engine arm")
    aa = sub.add_parser("aa")
    r = sub.add_parser("results")
    r.add_argument("--start", required=True)
    r.add_argument("--end", help="default: yesterday")
    for p in (a, aa, r, rf):
        p.add_argument("--shop", default="demo", choices=sorted(SHOPS))
    for p in (aa, r):
        p.add_argument("--front", help="storefront to read (shops with several fronts); default the first")
        p.add_argument("--all-visitors", action="store_true", help="ignore the shop's GA4 segment (guardrail read)")
    args = ap.parse_args()
    if args.cmd == "assign":
        assign(args.shop, args.run, args.top, args.skip, frozenset(int(x) for x in args.exclude.split(',') if x))
        return
    if args.cmd == "refresh":
        refresh(args.shop, args.run)
        return
    if args.cmd == "aa":
        end = dt.date.today() - dt.timedelta(days=1)
        res = compare(args.shop, end - dt.timedelta(days=PRE_DAYS - 1), end, args.front, not args.all_visitors)
        res["note"] = "A/A: nothing differs between the arms yet, so every lift should be near 0 and every CI cover 0."
    else:
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
        res = compare(args.shop, start, end, args.front, not args.all_visitors)
    tag = (f"_{args.front}" if args.front else "") + ("_all" if getattr(args, "all_visitors", False) else "")
    path = kit_dir(args.shop) / f"{args.cmd}{tag}_{dt.date.today()}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    show(res)


if __name__ == "__main__":
    main()
