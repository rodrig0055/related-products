"""Turn the raw inputs into one Product record per published giftable product."""

import html
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import COLLECTION_SKIP, COLOURS, CVR_PRIOR_SESSIONS, DESCRIPTION_KEEP, NAME_STOPWORDS
from .shops import Shop


@dataclass
class Product:
    id: int
    slug: str
    name: str
    url: str
    image: str | None
    price: float
    price_source: str                       # "ga4" (shown price, last 30 days) or "list"
    facets: dict[str, set[str]]             # facet -> subcategory slugs
    colours: set[str]
    name_tokens: set[str]
    views: int = 0                          # GA4 sessions that viewed the PDP (12 months)
    bought: int = 0                         # ...and bought it in the same session
    cvr: float = 0.0                        # smoothed toward the shop mean
    orders: int = 0                         # delivered orders, 12 months
    revenue: float = 0.0
    search_impressions: int = 0             # non-brand GSC impressions
    true_pr: float | None = None            # TIPR link equity
    exclusive: set[str] = field(default_factory=set)   # exclusive occasions it is tagged with
    exclusive_only: bool = False            # all its occasions are exclusive ones
    strict: set[str] = field(default_factory=set)       # strict-lane occasions it is tagged with
    seasonal_only: bool = False
    linkable: bool = True
    not_linkable_why: str = ""
    # Storefront key -> hubs where it is sold on that front (one front, one hub: {"demo": {1}}).
    fronts: dict[str, frozenset[int]] = field(default_factory=dict)
    cities: dict[str, frozenset[int]] = field(default_factory=dict)   # per front; absent = every city
    # Storefront key -> the hub a card must be sold in: where this product mostly sells.
    need_hub: dict[str, int] = field(default_factory=dict)
    lead: str | None = None                 # first front it is published on: cards must work there
    collections: set[str] = field(default_factory=set)   # themed collections (COLLECTION_SKIP removed)
    text: str = ""                          # name + opening + composition sections, for TF-IDF

    @property
    def subs(self) -> set[str]:
        return set().union(*self.facets.values()) if self.facets else set()


def _text(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).lower()


def colours_of(name: str, description: str | None) -> set[str]:
    """Colours named in the product name; the description only if the name names none."""
    def find(text):
        return {c for c, words in COLOURS.items() if any(re.search(rf"\b{re.escape(w)}s?\b", text) for w in words)}
    return find(_text(name)) or find(_text(description)[:400])


def description_text(name: str, desc: str | None) -> str:
    """What the product is: its name (twice, it is the densest signal), the opening paragraph, and
    the composition sections. Sections are split on <strong> headings; dimensions, packaging,
    notes, personal-message and care sections are left out (the same boilerplate on every product)."""
    parts = re.split(r"<strong[^>]*>(.*?)</strong>", desc or "", flags=re.S)
    intro, comp = [parts[0]], []
    for i in range(1, len(parts) - 1, 2):
        head = _text(parts[i]).strip(" :.…").replace("&hellip;", "")
        if any(head.startswith(k) for k in DESCRIPTION_KEEP):
            comp.append(parts[i + 1])
    # What is inside (the composition) counts double: it is what makes two products alike.
    return " ".join([_text(name)] * 2 + [_text(k) for k in intro] + [_text(k) for k in comp] * 2)


BLOOM_WORDS = re.compile(r"\b(roses?|tulips?|lil(y|ies)|carnations?|gerberas?|sunflowers?|orchids?|hydrangeas?|peon(y|ies)|"
                         r"lisianthus|chrysanthemums?|mums|daisies|daisy|stargazers?|eucalyptus|baby'?s breath|"
                         r"blooms?|bouquet|stems?|anthuriums?|calla|ranunculus|gypsophila|statice|aster)\b")


def name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _text(name)) if t not in NAME_STOPWORDS}


def need_hubs(shop: Shop, fronts: dict[str, frozenset[int]], by_hub: dict[int, int]) -> dict[str, int]:
    """Per front, the hub a card must be sold in: the hub this product has most orders in (among
    the hubs it is sold in on that front), else the biggest hub it is sold in."""
    rank = {h: i for i, h in enumerate(shop.hub_order)}
    out = {}
    for f, hubs in fronts.items():
        if hubs:
            out[f] = min(hubs, key=lambda h: (-by_hub.get(h, 0), rank.get(h, 99), h))
    return out


def build_products(shop: Shop, cat: dict, sales: dict, beh: dict, demand: dict, equity: dict,
                   hub_sales: dict | None = None, colls: dict | None = None) -> dict[int, Product]:
    skip = re.compile(COLLECTION_SKIP)
    subs_of: dict[int, dict[str, set[str]]] = {}
    sub_live: dict[str, bool] = {}
    for r in cat["subs"]:
        facet = shop.facet_overrides.get(r["sub"]) or shop.facets.get(r["cat"])
        if not facet:
            continue
        sub = shop.sub_aliases.get(r["sub"], r["sub"])
        subs_of.setdefault(int(r["product_id"]), {}).setdefault(facet, set()).add(sub)
        live = bool(int(r["sub_published"] or 0)) and bool(int(r["cat_published"] or 0))
        sub_live[sub] = sub_live.get(sub, False) or live

    gp = beh.get("products", {}) if beh else {}
    tot_views = sum(v["views"] for v in gp.values())
    p0 = (sum(v["bought"] for v in gp.values()) / tot_views) if tot_views else 0.02

    products: dict[int, Product] = {}
    for r in cat["products"]:
        pid = int(r["id"])
        facets = {f: set(s) for f, s in subs_of.get(pid, {}).items()}
        # Kind: a product with a flower-type tag is (at least) flowers; an untagged one is flowers
        # only if its name or description names blooms (a cash gift or a mug is not a bouquet).
        if facets.get("flower") or (not facets.get("kind") and BLOOM_WORDS.search(_text(r["name"] + " " + (r["description"] or "")))):
            facets.setdefault("kind", set()).add("flowers")
        g = gp.get(pid) or gp.get(str(pid)) or {}
        ga_price = g.get("price")
        p = Product(
            id=pid, slug=r["slug"], name=r["name"], url=f"{shop.origin}/product/{r['slug']}",
            image=(shop.image_base + cat["images"][pid]) if pid in cat["images"] else None,
            price=float(ga_price or r["list_price"]), price_source="ga4" if ga_price else "list",
            facets=facets, colours=colours_of(r["name"], r["description"]), name_tokens=name_tokens(r["name"]),
            views=int(g.get("views", 0)), bought=int(g.get("bought", 0)),
        )
        p.cvr = (p.bought + CVR_PRIOR_SESSIONS * p0) / (p.views + CVR_PRIOR_SESSIONS)
        fr = r.get("fronts") or {shop.fronts[0].key: list(shop.hub_ids or (0,))}
        p.fronts = {f.key: frozenset(fr[f.key]) for f in shop.fronts if fr.get(f.key)}
        p.need_hub = need_hubs(shop, p.fronts, (hub_sales or {}).get(pid, {}))
        p.cities = {f: frozenset(c) for f, c in (r.get("cities") or {}).items() if f in p.fronts}
        p.lead = next(iter(p.fronts), None)
        p.collections = {c for c in (colls or {}).get(pid, set()) if not skip.search(c.lower())}
        p.text = description_text(r["name"], r["description"])
        s = sales.get(pid, {})
        p.orders, p.revenue = int(s.get("orders", 0)), float(s.get("revenue", 0))
        p.search_impressions = int(demand.get(p.slug, {}).get("impressions", 0))
        p.true_pr = equity.get(p.slug)
        occ = facets.get("occasion", set())
        p.exclusive = occ & set(shop.exclusive)
        p.exclusive_only = bool(occ) and occ <= set(shop.exclusive)
        p.strict = occ & set(shop.strict)
        p.seasonal_only = bool(occ) and occ <= set(shop.seasonal)
        # "not on sale" first: the page itself is hidden, so it gets no cards either.
        if "fronts" in r and not r["fronts"]:
            p.linkable, p.not_linkable_why = False, "not on sale (no published vendor or variant)"
        elif pid in shop.blocklist:
            p.linkable, p.not_linkable_why = False, "blocklisted (test or internal product)"
        elif int(r["search_exclude"] or 0):
            p.linkable, p.not_linkable_why = False, "excluded from search"
        elif p.seasonal_only and not any(sub_live.get(s) for s in occ):
            p.linkable, p.not_linkable_why = False, f"out of season ({', '.join(sorted(occ))} unpublished)"
        products[pid] = p
    return products


def idf(products: dict[int, Product]) -> dict[str, float]:
    n = len(products)
    df = Counter(s for p in products.values() for s in p.subs)
    return {s: math.log((n + 1) / (c + 1)) + 0.1 for s, c in df.items()}
