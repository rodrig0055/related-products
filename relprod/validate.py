"""Is the test worth running? Four checks on past data, before anything is loaded.

  uv run python -m relprod.validate --shop demo

1. balance   Pairs stay as assigned; only the coin flips inside pairs are redrawn. Of N_SEEDS
             draws, keep the one whose arms differ least on the last 6 weeks (CUPED-adjusted, the
             same estimator as the real read). Done before any upload, never after results.
2. chance    Randomization inference: redraw the coin flips 400 times on two past windows where
             nothing differed; the 95% band of those "lifts" is what chance alone produces. The
             pair bootstrap was too narrow (a split balanced on one window showed +24% on the
             previous one, outside its own interval), so the read uses this band instead.
3. power     Add a known lift to the engine arm of random splits (extra related clicks, extra
             buying sessions drawn at random) and count how often it clears the chance band.
4. pre/post  What a plain "next 6 weeks vs previous 6" would have said last year over the same
             calendar weeks, when nothing changed: the season alone.

Writes out/<shop>/abtest/validate_<front>.json. `--apply` rewrites the assignment with the
chosen split (abtest.assign with that seed).
"""

import argparse
import datetime as dt
import json
import math
import random
from collections import defaultdict

from .abtest import METRICS, PRE_DAYS, _estimate, _pair_rows, assign, front_of, kit_dir, load_assignment, per_product
from .measure import flow
from .shops import SHOPS

N_SEEDS = 400
N_NULL = 400
BOOT = 400
LIFTS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
REPS = 40
BALANCE_ON = ("click_rate", "multi_pdp", "conversion", "revenue")


def windows(end: dt.date) -> dict[str, tuple[dt.date, dt.date]]:
    w = dt.timedelta(days=PRE_DAYS)
    a_end = end
    a_start = a_end - w + dt.timedelta(days=1)
    b_end = a_start - dt.timedelta(days=1)
    b_start = b_end - w + dt.timedelta(days=1)
    c_end = b_start - dt.timedelta(days=1)
    c_start = c_end - w + dt.timedelta(days=1)
    return {"A": (a_start, a_end), "B": (b_start, b_end), "C": (c_start, c_end)}


def _cap(rows):
    ws = sorted(v[2] for v in rows.values())
    return ws[int(0.9 * (len(ws) - 1))] if ws else 1


def lift(pairs, arm_of, test, pre, metric, boot=0, rng=None):
    tr, pr = _pair_rows(pairs, test, arm_of, metric), _pair_rows(pairs, pre, arm_of, metric)
    idx = list(tr)
    if not idx:
        return None, None
    cap = _cap(tr)
    point = _estimate(idx, tr, pr, cap)
    if not boot:
        return point, None
    bs = sorted(v for v in (_estimate([rng.choice(idx) for _ in idx], tr, pr, cap) for _ in range(boot)) if v is not None)
    return point, (bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1])


def arms_for(pair_list, seed):
    rng = random.Random(seed)
    arm = {}
    for a, b in pair_list:
        x, y = ("engine", "rotation") if rng.random() < 0.5 else ("rotation", "engine")
        arm[a], arm[b] = x, y
    return arm


def inject(data, engine, metric, l, rng):
    """Copy of per-product data with the engine arm's numerator raised by l (Poisson draws)."""
    num = METRICS[metric][0]
    out = {}
    for pid, x in data.items():
        y = dict(x)
        if pid in engine and y.get(num):
            if num in ("clicks", "multi", "buyers"):
                y[num] = x[num] + _poisson(rng, l * x[num])
                if num == "buyers":
                    y["revenue"] = x["revenue"] * (y[num] / x[num]) if x[num] else x["revenue"]
            else:
                y[num] = x[num] * (1 + l)
        out[pid] = y
    return out


def _poisson(rng, lam):
    if lam <= 0:
        return 0
    if lam > 50:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    k, p, L = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def front_rates(fl):
    views = sum(fl["views"].values())
    s = fl["sessions"]
    return {"pdp_views": views, "related_clicks_per_view": len(fl["clicks"]) / views if views else 0,
            "multi_pdp": sum(x["pdps"] > 1 for x in s) / len(s) if s else 0,
            "conversion": sum(x["bought"] for x in s) / len(s) if s else 0}


def run(shop_key: str, front_key: str | None, end: dt.date, apply: bool) -> dict:
    shop = SHOPS[shop_key]
    fr = front_of(shop, front_key)
    rows = load_assignment(shop_key)
    slug_id = {r["slug"]: r["product_id"] for r in rows}
    by_pair = defaultdict(list)
    for r in rows:
        if r["pair"] >= 0:
            by_pair[r["pair"]].append(r["product_id"])
    pair_ids = [tuple(v) for _, v in sorted(by_pair.items())]
    pairs = [list(p) for p in pair_ids]
    W = windows(end)
    d = {k: per_product(fr, s, e, slug_id) for k, (s, e) in W.items()}
    out = {"front": fr.name, "windows": {k: f"{s}..{e}" for k, (s, e) in W.items()}, "pairs": len(pairs)}

    # 1. balance: current split vs the best of N_SEEDS redraws, on window A (pre = B)
    current = {r["product_id"]: r["arm"] for r in rows}
    def imbalance(arm):
        return {m: lift(pairs, arm, d["A"], d["B"], m)[0] for m in BALANCE_ON}
    cur = imbalance(current)
    scored = []
    for seed in range(1, N_SEEDS + 1):
        arm = arms_for(pair_ids, seed)
        im = imbalance(arm)
        scored.append((max(abs(v) for v in im.values() if v is not None), seed, im))
    scored.sort()
    best_score, best_seed, best_im = scored[0]
    typical = sorted(s for s, _, _ in scored)[len(scored) // 2]
    out["balance"] = {"current": cur, "chosen_seed": best_seed, "chosen": best_im,
                      "worst_metric_current": max(abs(v) for v in cur.values() if v is not None),
                      "worst_metric_chosen": best_score, "worst_metric_typical_draw": typical}
    arm = arms_for(pair_ids, best_seed)

    # 2 + 3. Chance, measured with the test's own coin flips (randomization inference): redraw the
    # flips N_NULL times on real past data where nothing differed, and see how big a "lift" pure
    # chance produces. A real result has to beat the 95% band of that. Done on two windows (A with
    # pre B, B with pre C), because products' click rates move a lot from one 6 weeks to the next.
    draws = [arms_for(pair_ids, 10_000 + i) for i in range(N_NULL)]
    out["chance"] = {}
    for m in METRICS:
        band = {}
        for w, pre in (("A", "B"), ("B", "C")):
            vals = sorted(v for v in (lift(pairs, a_, d[w], d[pre], m)[0] for a_ in draws) if v is not None)
            if vals:   # a front with no click sales at all, so no band for that metric
                band[w] = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])
        if not band:
            continue
        width = max(max(abs(lo), abs(hi)) for lo, hi in band.values())
        out["chance"][m] = {"label": METRICS[m][2], "band_95": band, "needs_more_than": width}
    # The split the bootstrap judged balanced, on the window it was not chosen on: inside the band?
    out["holdout"] = {m: {"lift": lift(pairs, arm, d["B"], d["C"], m)[0], "chance_band": out["chance"][m]["band_95"].get("B")}
                      for m in out["chance"]}

    # Power: add a true lift to the engine arm of random splits in window A; detected when the
    # estimate clears the chance band. Share of splits that detect it.
    out["power"] = {}
    for m in ("click_rate", "multi_pdp", "conversion", "revenue"):
        hi = out["chance"][m]["needs_more_than"]
        res = {}
        metric = "conversion" if m == "revenue" else m
        for l in LIFTS:
            hits = 0
            for r in range(REPS):
                a_ = draws[r]
                eng = {pid for pid, x in a_.items() if x == "engine"}
                test = inject(d["A"], eng, metric, l, random.Random(1000 * r + int(l * 100)))
                v = lift(pairs, a_, test, d["B"], m)[0]
                hits += v is not None and v > hi
            res[str(l)] = hits / REPS
        out["power"][m] = {"label": METRICS[m][2], "detected_share_by_true_lift": res}

    # 4. pre/post on last year's same calendar weeks (all products on this front, nothing changed)
    ly = lambda x: x.replace(year=x.year - 1)
    a_s, a_e = W["A"]
    nxt = (ly(a_e) + dt.timedelta(days=1), ly(a_e) + dt.timedelta(days=PRE_DAYS))
    prv = (ly(a_s), ly(a_e))
    r_next, r_prev = front_rates(flow(fr, *nxt)), front_rates(flow(fr, *prv))
    out["pre_post_last_year"] = {"previous": f"{prv[0]}..{prv[1]}", "next": f"{nxt[0]}..{nxt[1]}",
                                 "change": {k: (r_next[k] / r_prev[k] - 1) if r_prev[k] else None for k in r_prev},
                                 "previous_rates": r_prev, "next_rates": r_next}
    path = kit_dir(shop_key) / f"validate_{fr.key}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    if apply:
        out["applied_seed"] = best_seed
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shop", default="demo", choices=sorted(SHOPS))
    ap.add_argument("--front")
    ap.add_argument("--end", help="last day of window A (default: yesterday)")
    a = ap.parse_args()
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=1)
    res = run(a.shop, a.front, end, False)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
