import json
"""Rule checks on a synthetic catalog. Run: uv run python -m tests.test_engine"""

from collections import Counter

from relprod import config as C
from relprod.engine import Engine, is_dup, same_family
from relprod.features import Product, build_products, name_tokens
from relprod.shops import Shop


def mk(pid, name, price, occ=(), rec=(), kind=("flowers",), flw=(), colours=(), views=100, **kw):
    facets = {"occasion": set(occ), "recipient": set(rec), "kind": set(kind), "flower": set(flw)}
    return Product(id=pid, slug=f"p{pid}", name=name, url="", image=None, price=price, price_source="list",
                   facets={k: v for k, v in facets.items() if v}, colours=set(colours),
                   name_tokens=name_tokens(name), views=views, cvr=0.03, **kw)


def catalog():
    ps = [
        mk(1, "Rosy Glow (12 Roses)", 59, ["anniversary", "romance"], flw=["rose"], colours=["pink"], views=5000),
        mk(2, "Grand Gesture (24 Roses)", 79, ["anniversary", "romance"], flw=["rose"], colours=["red"]),
        mk(3, "Forever Yours (30 Roses)", 99, ["anniversary", "romance"], flw=["rose"], colours=["red"]),
        mk(4, "Pink Tulips in Vase", 69, ["birthday", "romance"], flw=["tulip"], colours=["pink"]),
        mk(5, "Yellow Tulips in Vase", 69, ["birthday", "romance"], flw=["tulip"], colours=["yellow"]),
        mk(6, "Violet Tulips in Vase", 69, ["birthday", "romance"], flw=["tulip"], colours=["purple"]),
        mk(7, "Deepest Sympathy", 108, ["funeral"]),
        mk(8, "Heaven's Tribute", 148, ["funeral"]),
        mk(9, "Peace & Purity", 99, ["funeral"], flw=["lilies"]),
        mk(10, "White Tulips in Vase", 69, ["funeral", "just-because"], flw=["tulip"], colours=["white"]),
        mk(11, "Snow Globe Christmas Box", 59, ["christmas"], seasonal_only=True, linkable=False),
        mk(12, "Grand Gesture (24 Roses) Bundle", 99, ["anniversary", "romance"], flw=["rose"]),
        mk(13, "Lily", 69, ["birthday", "getwell"], flw=["lilies"], colours=["white"]),
        mk(14, "Sunny Day", 55, ["birthday", "getwell"], flw=["sunflower"], colours=["yellow"]),
        mk(15, "Chocolate Cake", 59, ["birthday"], kind=["cakes"]),
        mk(16, "Strawberry Cake", 65, ["birthday"], kind=["cakes"]),
        mk(17, "Delicate Dream", 79, ["anniversary", "graduation", "funeral"], flw=["tulip"], colours=["white"]),
        mk(18, "White Promise", 88, [], flw=["tulip", "rose"], colours=["white"]),
    ]
    for p in ps:
        p.exclusive = p.facets.get("occasion", set()) & {"funeral"}
        p.exclusive_only = bool(p.facets.get("occasion")) and p.facets["occasion"] <= {"funeral"}
        p.strict = p.facets.get("occasion", set()) & {"funeral"}
    return {p.id: p for p in ps}


def test_rules():
    P = catalog()
    picks, inbound = Engine(P, {}).run()
    for aid, chosen in picks.items():
        a, targets = P[aid], [P[c.target] for c in chosen]
        assert len({t.id for t in targets}) == len(targets), f"{a.name}: duplicate card"
        assert aid not in {t.id for t in targets}, f"{a.name}: links to itself"
        assert all(t.linkable for t in targets), f"{a.name}: unlinkable card"
        for t in targets:
            assert not is_dup(a, t), f"{a.name}: near-duplicate {t.name}"
            if a.exclusive_only:
                assert t.exclusive & a.exclusive, f"{a.name} (funeral) shows {t.name}"
            assert t.strict == a.strict, f"{a.name} and {t.name} are on different sides of the funeral lane"
            if t.exclusive_only:
                assert t.exclusive & a.exclusive, f"funeral-only {t.name} shown on {a.name}"
        fam = sum(same_family(a, t) for t in targets)
        assert fam <= 1, f"{a.name}: {fam} colour variants of itself"
    # Out-of-season product is never linked; Grand Gesture never next to its own bundle.
    assert inbound[11] == 0
    assert 12 not in {c.target for c in picks[2]} and 2 not in {c.target for c in picks[12]}
    # A step up is never cheaper, unless it is a labelled fallback.
    for aid, chosen in picks.items():
        for c in chosen:
            if c.slot == "step_up" and not c.reason.startswith("Fallback"):
                assert P[c.target].price > P[aid].price, f"{P[aid].name}: step up {P[c.target].name} is cheaper"
    print("rules ok")


def test_kind_and_season_from_catalog():
    shop = Shop(key="t", name="T", origin="https://t", currency="X", product_type_ids=(1,), facets={"occ": "occasion", "flw": "flower", "g": "kind"},
                exclusive=("funeral",), seasonal=("christmas",))
    cat = {"products": [
        {"id": 1, "slug": "a", "name": "Red Roses", "description": "", "search_exclude": 0, "list_price": 50},
        {"id": 2, "slug": "b", "name": "Xmas Box", "description": "", "search_exclude": 0, "list_price": 50},
        {"id": 3, "slug": "c", "name": "Cake", "description": "a white cake", "search_exclude": 0, "list_price": 50},
    ], "subs": [
        {"product_id": 1, "sub": "rose", "sub_published": 1, "cat": "flw", "cat_published": 1},
        {"product_id": 2, "sub": "christmas", "sub_published": 0, "cat": "occ", "cat_published": 1},
        {"product_id": 3, "sub": "cakes", "sub_published": 1, "cat": "g", "cat_published": 1},
    ], "images": {}}
    P = build_products(shop, cat, {}, {}, {}, {})
    assert P[1].facets["kind"] == {"flowers"} and P[1].colours == {"red"}
    assert not P[2].linkable and "out of season" in P[2].not_linkable_why
    assert P[3].facets["kind"] == {"cakes"} and P[3].colours == {"white"}
    print("features ok")


def check_output(path):
    """Invariants on a real run: uv run python -m tests.test_engine out/demo/<date>"""
    import csv
    rows = list(csv.DictReader(open(f"{path}/related.csv")))
    prods = {r["id"]: r for r in csv.DictReader(open(f"{path}/products.csv"))}
    per = Counter(r["source_id"] for r in rows)
    # Every PDP shows 4 cards: price and relevance loosen (run, step 6).
    # Exception: a funeral or grand-opening product with too few lane-mates sold in all its hubs
    # and cities stays short (short beats an off-lane or unavailable card).
    from relprod.shops import SHOPS
    from pathlib import Path as _P
    shop = SHOPS.get(_P(path).resolve().parent.name)
    lanes = set(shop.strict) | set(shop.exclusive) if shop else set()
    sm = _P(path) / "summary.json"
    kept = {str(i) for i in json.loads(sm.read_text()).get("kept_live", [])} if sm.exists() else set()
    short = [prods[pid]["name"] for pid in prods if per.get(pid, 0) < C.CARDS
             and not prods[pid]["not_linkable_why"].startswith("not on sale")   # page not shown
             and not (set(prods[pid]["occasions"].split()) & lanes)
             and pid not in kept]   # live sets left as they are (build --keep-live)
    assert not short, f"{len(short)} PDPs have fewer than 4 cards: {short[:5]}"
    # No duplicates of any kind: no card twice, no self link, no position twice,
    # and no near-duplicate names (sizes of one product) in a row or of the product itself.
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source_id"], []).append(r)
    for sid, rs in by_src.items():
        ids = [r["related_id"] for r in rs]
        assert len(set(ids)) == len(ids), f"{prods[sid]['name']}: a card twice"
        assert sid not in ids, f"{prods[sid]['name']} recommends itself"
        assert len({r["position"] for r in rs}) == len(rs), f"{prods[sid]['name']}: a position twice"
        toks = [name_tokens(prods[i]["name"]) for i in [sid] + ids]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                a = Product(id=0, slug="", name="", url="", image=None, price=1, price_source="", facets={},
                            colours=set(), name_tokens=toks[i])
                b = Product(id=1, slug="", name="", url="", image=None, price=1, price_source="", facets={},
                            colours=set(), name_tokens=toks[j])
                assert not is_dup(a, b), f"{prods[sid]['name']}: near-duplicate names in its row"
    inb = Counter(r["related_id"] for r in rows)
    under = []
    for pid, p in prods.items():
        if p["linkable"] == "True":
            assert inb[pid] <= C.MAX_INBOUND, f"{p['name']}: {inb[pid]} inbound"
            if inb[pid] < C.MIN_INBOUND:
                under.append(p["name"])
        else:
            assert inb[pid] == 0, f"unlinkable {p['name']} is linked"
    # A single-hub shop reaches MIN_INBOUND for every product. Multi-hub shops may not: hubs and
    # cities are a hard rule, so a product sold in a few cities can only sit on pages limited to
    # those same cities, and most of those are not relevant to it. Allow under 5%.
    assert len(under) <= len(prods) * 5 // 100, f"{len(under)} products below {C.MIN_INBOUND} inbound: {under[:5]}"
    print(f"output ok: {len(per)} PDPs x {C.CARDS}, inbound {min(inb.values())}..{max(inb.values())}")


def test_feedback():
    """A card that drew fewer clicks than its position's average (factor < 1) gives way to an
    equally relevant card nobody has tried yet. Price bands still come first."""
    ps = [mk(1, "Rosy Glow", 60, ["anniversary"], flw=["rose"], colours=["pink"], views=900),
          mk(2, "Crimson Promise", 61, ["anniversary"], flw=["rose"], colours=["red"]),
          mk(3, "Scarlet Vow", 62, ["anniversary"], flw=["rose"], colours=["red"])]
    ps += [mk(10 + i, f"Filler {w}", 60, ["birthday"], kind=["cakes"]) for i, w in enumerate("abcdef")]
    P = {p.id: p for p in ps}
    first = Engine(P, {}).run()[0][1][0]
    fb = {"cards": [{"a": 1, "b": first.target, "factor": 0.5}]}
    again = Engine(P, {}, fb).run()[0][1][0]
    assert again.slot == first.slot and again.target != first.target, "loser card kept its slot"
    print("feedback ok")


def test_hubs_and_minimum():
    """A card must be on sale in every hub the product is (hard rule: the storefront hides it
    elsewhere), and every PDP still shows 4 cards by loosening price and relevance, never lanes."""
    ps = [mk(1, "Downtown Roses", 60, ["anniversary"], flw=["rose"], views=900),
          mk(2, "Everywhere Roses", 62, ["anniversary"], flw=["rose"]),
          mk(3, "Scarlet Dawn", 61, ["anniversary"], flw=["rose"]),
          mk(4, "Lone Orchid", 80, ["anniversary"], kind=["plants"])]
    ps += [mk(10 + i, f"{w} Cake", 60, ["birthday"], kind=["cakes"])
           for i, w in enumerate(["Mango", "Chocolate", "Pandan", "Coconut", "Matcha", "Lemon"])]
    ps += [mk(20, "Wreath Peace", 3000, ["funeral"]), mk(21, "Wreath Calm", 3000, ["funeral"])]
    for p in ps:
        p.exclusive = p.facets.get("occasion", set()) & {"funeral"}
        p.strict = set(p.exclusive)
        p.exclusive_only = bool(p.exclusive)
        p.fronts = {"x": frozenset({1, 2})}
        p.need_hub = {"x": 1}
        p.lead = "x"
    ps[2].fronts = {"x": frozenset({1})}
    P = {p.id: p for p in ps}
    P[10].cities = {"x": frozenset({101})}        # Mango Cake: one city only
    P[11].cities = {"x": frozenset({101, 102})}   # Chocolate Cake: two cities
    picks, _ = Engine(P, {}).run()
    assert 3 not in {c.target for c in picks[1]}, "card missing from one of the product's hubs"
    assert all(10 not in {c.target for c in picks[a]} for a in P if a != 10), "city-limited card shown where it is not sold"
    assert 11 in {c.target for c in picks[10]} or len(picks[10]) == C.CARDS
    assert 2 in {c.target for c in picks[1]}
    for aid, chosen in picks.items():
        # Only a lane too small to hold 4 others stays short (2 wreaths here).
        want = C.CARDS if not P[aid].strict else min(C.CARDS, sum(1 for x in ps if x.strict) - 1)
        assert len(chosen) == want, f"{P[aid].name}: {len(chosen)} cards"
    assert {c.target for c in picks[20]} == {21}, "wreath lane broken"
    print("hubs ok")


if __name__ == "__main__":
    import sys
    test_hubs_and_minimum()
    test_rules()
    test_feedback()
    test_kind_and_season_from_catalog()
    if len(sys.argv) > 1:
        check_output(sys.argv[1])
