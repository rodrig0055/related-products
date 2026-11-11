"""Generate a synthetic shop for the demo: a fictional florist and gift shop, Acme Florals.

  python examples/make_demo.py            # writes cache/demo/*.json
  python run.py build --shop demo

Nothing here is real data. The products are invented, and shopper behaviour comes from a small
simulation: each session has an occasion and a budget, views a few products that fit it, and
sometimes buys one. That is enough for co-views, "bought instead" pairs and conversion rates to
carry the same kind of signal a real GA4 export does.
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "cache" / "demo"
rng = random.Random(7)

# Demo photos from Unsplash, by what they show (downloaded only when the report is built).
IMG = {
    "roses": "photo-1508610048659-a06b669e3321",
    "tulips": "photo-1468327768560-75b778cbb551",
    "tulips-vase": "photo-1561181286-d3fee7d55364",
    "sunflowers": "photo-1455659817273-f96807779a8a",
    "peonies": "photo-1563241527-3004b7be0ffd",
    "heart": "photo-1526047932273-341f2a7631f9",
    "gift-box": "photo-1549465220-1a8b9238cd48",
    "cake": "photo-1578985545062-69928b1d9587",
    "shop": "photo-1487070183336-b863922373d4",
    "yellow": "photo-1490750967868-88aa4486c946",
}
Q = "?w=400&h=400&fit=crop&q=70"

COLOURS = {"red": "Crimson", "pink": "Blush", "white": "Ivory", "yellow": "Sunny", "purple": "Lavender",
           "orange": "Peach"}
FLOWERS = {
    "roses": ("Rose", ["red", "pink", "white", "yellow", "orange"], 55, "roses"),
    "tulips": ("Tulip", ["pink", "white", "yellow", "purple", "red"], 45, "tulips"),
    "sunflowers": ("Sunflower", ["yellow"], 40, "sunflowers"),
    "peonies": ("Peony", ["pink", "white"], 75, "peonies"),
    "lilies": ("Lily", ["white", "pink", "orange"], 50, "shop"),
    "orchids": ("Orchid", ["white", "purple"], 70, "shop"),
    "gerberas": ("Gerbera", ["orange", "pink", "yellow"], 35, "yellow"),
    "carnations": ("Carnation", ["red", "pink", "white"], 30, "heart"),
}
FORMS = [("Bouquet", 1.0), ("Hand-tied Posy", 0.7), ("Grand Bouquet", 1.8), ("Vase Arrangement", 1.3)]
OCCASIONS = {"roses": ["anniversary", "love-and-romance", "birthday"],
             "tulips": ["birthday", "thank-you", "congratulations"],
             "sunflowers": ["birthday", "get-well", "congratulations", "thank-you"],
             "peonies": ["anniversary", "birthday", "love-and-romance"],
             "lilies": ["birthday", "get-well", "anniversary"],
             "orchids": ["congratulations", "thank-you", "anniversary"],
             "gerberas": ["birthday", "get-well", "thank-you"],
             "carnations": ["birthday", "thank-you", "get-well"]}
RECIPIENTS = ["for-her", "for-mom", "for-friends", "for-him"]
FILLERS = ["eucalyptus", "baby's breath", "ruscus", "statice", "pittosporum", "wax flower"]
WRAPS = ["kraft paper", "matte black wrap", "linen wrap", "a glass vase", "a ceramic pot", "tissue and ribbon"]

GIFTS = [  # name, kind, price, occasions, recipients, image, inside
    ("Chocolate Drip Cake", "cakes", 48, ["birthday", "congratulations"], ["for-friends"], "cake", "dark chocolate sponge, ganache drip, buttercream rosettes"),
    ("Strawberry Shortcake", "cakes", 42, ["birthday"], ["for-her", "for-friends"], "cake", "vanilla chiffon, fresh strawberries, whipped cream"),
    ("Salted Caramel Cake", "cakes", 46, ["birthday", "thank-you"], ["for-him", "for-friends"], "cake", "caramel sponge, salted caramel drip, toffee shards"),
    ("Red Velvet Celebration Cake", "cakes", 52, ["birthday", "anniversary"], ["for-her"], "cake", "red velvet sponge, cream cheese frosting"),
    ("Lemon Drizzle Loaf", "cakes", 28, ["thank-you", "get-well"], ["for-mom", "for-friends"], "cake", "lemon loaf, sugar glaze, candied peel"),
    ("Truffle Gift Box", "chocolates", 32, ["thank-you", "birthday", "love-and-romance"], ["for-her", "for-him"], "gift-box", "16 handmade truffles in four flavours"),
    ("Celebration Gift Box", "hampers", 65, ["birthday", "congratulations"], ["for-her", "for-friends"], "gift-box", "sparkling juice, truffles, scented candle"),
    ("Get Well Care Hamper", "hampers", 58, ["get-well"], ["for-mom", "for-friends"], "gift-box", "herbal tea, honey, dried fruit, soft socks"),
    ("Gourmet Snack Hamper", "hampers", 85, ["thank-you", "congratulations"], ["for-him", "for-friends"], "gift-box", "crackers, cheese straws, nuts, chutney"),
    ("Pamper Me Spa Box", "hampers", 72, ["birthday", "thank-you"], ["for-her", "for-mom"], "gift-box", "bath salts, body lotion, candle, eye mask"),
    ("Peace Lily Plant", "plants", 38, ["get-well", "sympathy", "thank-you"], ["for-mom"], "shop", "peace lily in a ceramic pot"),
    ("Monstera in Woven Basket", "plants", 55, ["congratulations", "thank-you"], ["for-him", "for-friends"], "shop", "monstera deliciosa, woven basket"),
    ("Mini Succulent Trio", "plants", 26, ["thank-you", "birthday"], ["for-friends"], "shop", "three succulents in terracotta pots"),
    ("Happy Birthday Balloon Bundle", "balloons", 30, ["birthday"], ["for-friends"], "heart", "five helium balloons with ribbons"),
    ("Heart Balloon Bouquet", "balloons", 34, ["love-and-romance", "anniversary"], ["for-her"], "heart", "heart-shaped foil balloons"),
]
SYMPATHY = [
    ("Serenity White Lily Wreath", 140, "white lilies, white roses, chrysanthemums"),
    ("Peaceful Standing Spray", 180, "white lilies, carnations, gladioli on an easel"),
    ("Gentle Remembrance Basket", 95, "white roses, lilies and orchids in a basket"),
    ("Eternal Rest Casket Spray", 220, "white roses, lilies, ferns"),
    ("Sympathy Orchid Arrangement", 85, "white phalaenopsis orchids, moss"),
]
OPENING = [
    ("Prosperity Flower Stand", 160, "sunflowers, red anthuriums, gerberas on a two-tier stand"),
    ("Grand Success Flower Stand", 220, "orchids, sunflowers, roses on a three-tier stand"),
    ("Golden Fortune Opening Basket", 120, "sunflowers, yellow gerberas, ribbons"),
]
SEASONAL = [  # name, occasion, price, image, inside
    ("Valentine's Dozen Red Roses", "valentines", 89, "roses", "12 long-stem red roses, red wrap"),
    ("Be Mine Rose and Truffle Set", "valentines", 99, "heart", "6 red roses, truffle box"),
    ("Mother's Day Pink Peony Jar", "mothers-day", 79, "peonies", "pink peonies, garden roses, glass jar"),
    ("Mum's Garden Basket", "mothers-day", 69, "tulips-vase", "pink tulips, carnations, wicker basket"),
    ("Christmas Pine and Rose Centrepiece", "christmas", 75, "shop", "red roses, pine, pinecones, candle"),
    ("Festive Hamper", "christmas", 95, "gift-box", "fruit cake, cookies, mulled spice, wine glasses"),
    ("Winter Berry Wreath", "christmas", 60, "shop", "pine, holly berries, eucalyptus"),
]


def slugify(s: str) -> str:
    s = "".join(c if c.isalnum() else "-" for c in s.lower().replace("'", ""))
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def flower_products():
    out = []
    for key, (noun, colours, base, img) in FLOWERS.items():
        for colour in colours:
            forms = rng.sample(FORMS, k=3 if key in ("roses", "tulips") else 2)
            for form, mult in forms:
                n = rng.choice([6, 10, 12, 15, 20, 24])
                name = f"{COLOURS[colour]} {noun} {form}"
                inside = f"{n} {colour} {key}, {rng.choice(FILLERS)}, {rng.choice(FILLERS)}"
                occ = rng.sample(OCCASIONS[key], k=2)
                if colour == "red" and key == "roses":
                    occ = ["anniversary", "love-and-romance"]
                rec = rng.sample(RECIPIENTS, k=rng.choice([1, 2]))
                image = img
                if key == "tulips" and form == "Vase Arrangement":
                    image = "tulips-vase"
                if key == "roses" and colour in ("pink", "white"):
                    image = rng.choice(["roses", "heart", "peonies"])
                out.append(dict(name=name, kind="flowers", flower=key, colour=colour, occ=occ, rec=rec,
                                price=round(base * mult * rng.uniform(0.9, 1.15)), image=image,
                                inside=inside, wrap=rng.choice(WRAPS)))
    return out


def build():
    items = flower_products()
    for name, kind, price, occ, rec, img, inside in GIFTS:
        items.append(dict(name=name, kind=kind, flower=None, colour=None, occ=occ, rec=rec, price=price,
                          image=img, inside=inside, wrap="a gift box"))
    for name, price, inside in SYMPATHY:
        items.append(dict(name=name, kind="flowers", flower="lilies", colour="white", occ=["sympathy"], rec=[],
                          price=price, image="shop", inside=inside, wrap="a stand"))
    for name, price, inside in OPENING:
        items.append(dict(name=name, kind="flowers", flower="sunflowers", colour="yellow", occ=["grand-opening"],
                          rec=[], price=price, image="sunflowers" if "Basket" in name else "yellow", inside=inside,
                          wrap="a stand"))
    for name, occ, price, img, inside in SEASONAL:
        items.append(dict(name=name, kind="flowers" if "Hamper" not in name else "hampers", flower=None, colour=None,
                          occ=[occ], rec=["for-mom"] if occ == "mothers-day" else [], price=price, image=img,
                          inside=inside, wrap="seasonal wrap"))

    products, subs, images, colls = [], [], {}, []
    seen = Counter()
    for i, it in enumerate(items):
        pid = 101 + i
        slug = slugify(it["name"])
        seen[slug] += 1
        if seen[slug] > 1:
            slug += f"-{seen[slug]}"
        it.update(id=pid, slug=slug, quality=math.exp(rng.gauss(0, 0.35)), pop=math.exp(rng.gauss(0, 0.8)))
        desc = (f"<p>{it['name']}: {it['inside'].split(',')[0]}, finished in {it['wrap']}. "
                f"A thoughtful way to say it, delivered same day.</p>"
                f"<strong>Blooms:</strong> <p>{it['inside']}</p>" if it["kind"] == "flowers" else
                f"<p>{it['name']}, packed to travel and delivered same day.</p>"
                f"<strong>What's inside:</strong> <p>{it['inside']}</p>")
        desc += "<strong>Dimensions:</strong> <p>approx. 40 x 30 cm</p><strong>Care:</strong> <p>Keep cool, change water daily.</p>"
        products.append({"id": pid, "slug": slug, "name": it["name"], "description": desc, "product_type_id": 1,
                         "search_exclude": 0, "advance_days": 0, "list_price": it["price"], "variants": 1})
        images[pid] = IMG[it["image"]] + Q
        for o in it["occ"]:
            seasonal = o in ("valentines", "mothers-day", "christmas")
            published = 0 if o in ("valentines", "mothers-day") else 1   # out of season today
            subs.append({"product_id": pid, "sub": o, "sub_published": published if seasonal else 1,
                         "cat": "occasions", "cat_published": 1})
        for r in it["rec"]:
            subs.append({"product_id": pid, "sub": r, "sub_published": 1, "cat": "recipients", "cat_published": 1})
        if it["flower"]:
            subs.append({"product_id": pid, "sub": it["flower"], "sub_published": 1, "cat": "flowers", "cat_published": 1})
        if it["kind"] != "flowers":
            subs.append({"product_id": pid, "sub": it["kind"], "sub_published": 1, "cat": "gifts", "cat_published": 1})
        if it["colour"]:
            colls.append({"product_id": pid, "slug": f"{it['colour']}-flowers"})
        if it["price"] >= 90:
            colls.append({"product_id": pid, "slug": "luxury-gifts"})
        colls.append({"product_id": pid, "slug": "best-sellers" if it["pop"] > 1.5 else "under-50-gifts"})
    products[-1]["search_exclude"] = 0
    products[5]["search_exclude"] = 1      # one hidden product, to show the "not linkable" reasons
    return items, {"products": products, "subs": subs, "images": images}, colls


def simulate(items, sessions=60000):
    """Sessions with an occasion and a budget view 1-6 fitting products and sometimes buy one."""
    weights = {"birthday": 30, "anniversary": 14, "love-and-romance": 12, "get-well": 8, "congratulations": 9,
               "thank-you": 8, "sympathy": 6, "grand-opening": 3, "christmas": 4, "valentines": 3, "mothers-day": 3}
    occs, ws = list(weights), list(weights.values())
    by_occ = {o: [it for it in items if o in it["occ"]] for o in occs}
    views, bought, coview, subst = Counter(), Counter(), Counter(), Counter()
    for _ in range(sessions):
        occ = rng.choices(occs, ws)[0]
        pool = by_occ[occ] if rng.random() > 0.15 else items
        budget = math.exp(rng.gauss(math.log(60 if occ not in ("sympathy", "grand-opening") else 150), 0.45))
        fit = [it["pop"] * math.exp(-2.2 * abs(math.log(it["price"] / budget))) for it in pool]
        k = min(len(pool), 1 + min(5, int(rng.expovariate(0.55))))
        seen = []
        while len(seen) < k:
            it = rng.choices(pool, fit)[0]
            if it not in seen:
                seen.append(it)
        for it in seen:
            views[it["id"]] += 1
        for x in seen:
            for y in seen:
                if x["id"] < y["id"]:
                    coview[(x["id"], y["id"])] += 1
        if rng.random() < 0.075:
            pick = rng.choices(seen, [it["quality"] * math.exp(-1.5 * abs(math.log(it["price"] / budget))) for it in seen])[0]
            bought[pick["id"]] += 1
            for x in seen:
                if x is not pick:
                    subst[(x["id"], pick["id"])] += 1
    rows = [{"kind": "product", "a": it["id"], "b": None, "views": views[it["id"]], "bought": bought[it["id"]],
             "price": float(it["price"])} for it in items if views[it["id"]]]
    rows += [{"kind": "coview", "a": a, "b": b, "views": n, "bought": None, "price": None}
             for (a, b), n in coview.items() if n >= 2]
    rows += [{"kind": "subst", "a": a, "b": b, "views": n, "bought": None, "price": None} for (a, b), n in subst.items()]
    return {"start": "2025-11-11", "end": "2026-11-10", "rows": rows}, bought


def main():
    items, catalog, colls = build()
    ga4, bought = simulate(items)
    sales = [{"product_id": it["id"], "orders": round(bought[it["id"]] * 1.3), "revenue": round(bought[it["id"]] * 1.3 * it["price"], 2)}
             for it in items if bought[it["id"]]]
    gsc = [{"url": f"https://acmeflorals.example/product/{it['slug']}",
            "impressions": int(it["pop"] * rng.uniform(50, 2500)), "clicks": int(it["pop"] * rng.uniform(1, 40))}
           for it in items if rng.random() < 0.8]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in {"catalog.json": catalog, "collections.json": colls, "sales.json": sales,
                       "ga4.json": ga4, "gsc.json": gsc}.items():
        (OUT / name).write_text(json.dumps(data, indent=1))
    print(f"{len(items)} products, {len(ga4['rows'])} GA4 rows -> {OUT}")


if __name__ == "__main__":
    main()
