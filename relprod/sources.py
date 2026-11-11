"""Inputs: catalog, sales and collections (files you export from your shop), shopper behaviour
(GA4 BigQuery export), search demand (Search Console bulk export), link equity (a TIPR run).

Everything is read from cache/<shop>/. The catalog is required; every other input is optional: when
it is missing the engine ranks without it and the report says which inputs were used. GA4 and GSC
can be pulled straight from BigQuery with --refresh when the shop names its export tables.

File contract (all JSON, see examples/make_demo.py for a generator):

  catalog.json      {"products": [{id, slug, name, description, product_type_id, search_exclude,
                                   advance_days, list_price, variants, fronts?, cities?}],
                     "subs": [{product_id, sub, sub_published, cat, cat_published}],
                     "images": {product_id: path}}
  sales.json        [{product_id, orders, revenue}]                 last 12 months, delivered
  collections.json  [{product_id, slug}]
  hub_sales.json    [{product_id, hub_id, orders}]                  multi-hub shops only
  ga4.json          {start, end, rows: [{kind: product|coview|subst, a, b, views, bought, price}]}
  gsc.json          [{url, impressions, clicks}]                    non-brand, /product/<slug> URLs
  current_related.json  {product_id: [related ids in display order]}  what is live now (optional)
"""

import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from .shops import ROOT, Shop

CACHE = ROOT / "cache"


def env(key: str) -> str | None:
    local = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
    return os.environ.get(key) or local.get(key)


def _cached(shop: Shop, name: str, refresh: bool, fn):
    path = CACHE / shop.key / name
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str))
    return data


def _file(shop: Shop, name: str):
    """A file you export from your shop. There is no live connector for these: --refresh re-reads it."""
    path = CACHE / shop.key / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text())


def _has(shop: Shop, prefix: str) -> bool:
    return any((CACHE / shop.key).glob(f"{prefix}*.json"))


# --- Catalog (required) -------------------------------------------------------------------

def catalog(shop: Shop, refresh: bool = False) -> dict:
    """Published products of the giftable types, with list price, subcategories and main image.

    `fronts` (optional) says which storefronts sell a product and in which hubs; absent, it is sold
    on the shop's only front in every hub. An empty `fronts` means not on sale anywhere. A
    subcategory or category counts as published when any front publishes it.

    The list price is a fallback only: the storefront often shows a campaign price, so the engine
    prefers the price GA4 saw in the last 30 days.
    """
    try:
        data = _file(shop, "catalog.json")
    except FileNotFoundError as e:
        sys.exit(f"{e}. Export your catalog there (see the docstring of relprod/sources.py) "
                 f"or run: python examples/make_demo.py")
    data["images"] = {int(k): v for k, v in data.get("images", {}).items()}
    return data


def hub_sales(shop: Shop, refresh: bool = False) -> dict[int, dict[int, int]]:
    """Delivered orders per product per hub, for multi-hub shops (where a product sells)."""
    if shop.hub_ids:
        return {}
    try:
        rows = _file(shop, "hub_sales.json")
    except FileNotFoundError:
        print("hub sales: none, hubs ranked by size", file=sys.stderr)
        return {}
    out: dict[int, dict[int, int]] = {}
    for r in rows:
        out.setdefault(int(r["product_id"]), {})[int(r["hub_id"])] = int(r["orders"])
    return out


def collections(shop: Shop, refresh: bool = False) -> dict[int, set[str]]:
    """Collection slugs per product (any publish state: old campaign collections still say what a
    product is). Which collections count toward similarity is decided in features.py."""
    try:
        rows = _file(shop, "collections.json")
    except FileNotFoundError:
        print("collections: none, similarity without collections", file=sys.stderr)
        return {}
    out: dict[int, set[str]] = {}
    for r in rows:
        out.setdefault(int(r["product_id"]), set()).add(r["slug"])
    return out


# --- What is configured now (product_related) ------------------------------------------------

def current_related(shop: Shop) -> dict[int, list[int]] | None:
    """{product_id: [related ids in display order]} as live on the storefront, so a run can diff
    against what shoppers see. None when you have not exported it."""
    try:
        data = _file(shop, "current_related.json")
    except FileNotFoundError:
        return None
    return {int(k): [int(x) for x in v] for k, v in data.items()}


# --- Sales ------------------------------------------------------------------------------------

def sales(shop: Shop, refresh: bool = False) -> dict[int, dict]:
    """Delivered orders and revenue per product over the last 12 months."""
    try:
        rows = _file(shop, "sales.json")
    except FileNotFoundError:
        print("sales: none, ranking without sales", file=sys.stderr)
        return {}
    return {int(r["product_id"]): r for r in rows}


# --- Behaviour (GA4 export) ----------------------------------------------------------------

# A session is user_pseudo_id + ga_session_id. Sessions that view more than MAX_SESSION_VIEWS
# products are crawlers or scrapers, not shoppers, and are left out of the pair counts.
MAX_SESSION_VIEWS = 40

GA4_SQL = """
WITH ev AS (
  SELECT CONCAT(user_pseudo_id, '.', CAST((SELECT value.int_value FROM UNNEST(event_params)
                WHERE key = 'ga_session_id') AS STRING)) AS s,
         event_name, PARSE_DATE('%Y%m%d', event_date) AS d, SAFE_CAST(i.item_id AS INT64) AS item, i.price
  FROM `{table}`, UNNEST(items) i
  WHERE _TABLE_SUFFIX BETWEEN @start AND @end AND event_name IN ('view_item', 'purchase') {segment}
),
v AS (SELECT DISTINCT s, item FROM ev WHERE event_name = 'view_item' AND item IS NOT NULL AND s IS NOT NULL),
b AS (SELECT DISTINCT s, item FROM ev WHERE event_name = 'purchase' AND item IS NOT NULL AND s IS NOT NULL),
shoppers AS (SELECT s FROM v GROUP BY s HAVING COUNT(*) <= @max_views),
shown AS (
  SELECT item, APPROX_QUANTILES(price, 2)[OFFSET(1)] AS price FROM ev
  WHERE event_name = 'view_item' AND price > 0 AND d >= DATE_SUB(PARSE_DATE('%Y%m%d', @end), INTERVAL 30 DAY)
  GROUP BY item
)
SELECT 'product' AS kind, v.item AS a, CAST(NULL AS INT64) AS b,
       COUNT(*) AS views, COUNTIF(b.item IS NOT NULL) AS bought, ANY_VALUE(shown.price) AS price
FROM v LEFT JOIN b USING (s, item) LEFT JOIN shown ON shown.item = v.item GROUP BY v.item
UNION ALL
SELECT 'coview', x.item, y.item, COUNT(*), NULL, NULL
FROM v x JOIN v y ON x.s = y.s AND x.item < y.item JOIN shoppers ON shoppers.s = x.s
GROUP BY x.item, y.item HAVING COUNT(*) >= 2
UNION ALL
SELECT 'subst', x.item, y.item, COUNT(*), NULL, NULL
FROM v x JOIN b y ON x.s = y.s AND x.item != y.item JOIN shoppers ON shoppers.s = x.s
LEFT JOIN b own ON own.s = x.s AND own.item = x.item
WHERE own.item IS NULL
GROUP BY x.item, y.item
"""


# The same output from page views, for fronts or windows without view_item events (some
# storefronts never send it, or only started recently). The PDP slug is mapped to the product id
# with the catalog (@slugs / @ids), purchases keep their item ids.
GA4_PV_SQL = """
WITH ids AS (SELECT slug, ids[OFFSET(o)] AS item FROM UNNEST(@slugs) slug WITH OFFSET o),
ev AS (
  SELECT CONCAT(user_pseudo_id, '.', CAST((SELECT value.int_value FROM UNNEST(event_params)
                WHERE key = 'ga_session_id') AS STRING)) AS s,
         event_name,
         REGEXP_EXTRACT((SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location'), @pdp_re) AS slug,
         items
  FROM `{table}`
  WHERE _TABLE_SUFFIX BETWEEN @start AND @end AND event_name IN ('page_view', 'purchase') {segment}
),
v AS (SELECT DISTINCT ev.s, ids.item FROM ev JOIN ids USING (slug) WHERE ev.event_name = 'page_view' AND ev.s IS NOT NULL),
b AS (SELECT DISTINCT ev.s, SAFE_CAST(i.item_id AS INT64) AS item FROM ev, UNNEST(ev.items) i
      WHERE ev.event_name = 'purchase' AND ev.s IS NOT NULL AND SAFE_CAST(i.item_id AS INT64) IS NOT NULL),
shoppers AS (SELECT s FROM v GROUP BY s HAVING COUNT(*) <= @max_views)
SELECT 'product' AS kind, v.item AS a, CAST(NULL AS INT64) AS b,
       COUNT(*) AS views, COUNTIF(b.item IS NOT NULL) AS bought, CAST(NULL AS FLOAT64) AS price
FROM v LEFT JOIN b USING (s, item) GROUP BY v.item
UNION ALL
SELECT 'coview', x.item, y.item, COUNT(*), NULL, NULL
FROM v x JOIN v y ON x.s = y.s AND x.item < y.item JOIN shoppers ON shoppers.s = x.s
GROUP BY x.item, y.item HAVING COUNT(*) >= 2
UNION ALL
SELECT 'subst', x.item, y.item, COUNT(*), NULL, NULL
FROM v x JOIN b y ON x.s = y.s AND x.item != y.item JOIN shoppers ON shoppers.s = x.s
LEFT JOIN b own ON own.s = x.s AND own.item = x.item
WHERE own.item IS NULL
GROUP BY x.item, y.item
"""


def behaviour(shop: Shop, refresh: bool = False, days: int = 365, end: dt.date | None = None,
              pageviews: bool = False) -> dict:
    """Per product: sessions viewing it, sessions that viewed AND bought it, the price shown in the
    last 30 days. Per pair: sessions viewing both (coview), and sessions that viewed A, did not buy
    it, and bought B instead (subst, directed A -> B).

    Shops with several storefronts sum them (session counts add up; the price is the first
    front's). Fronts without view_item events are read from page views, and so is
    every front when `pageviews` is set (seasonal windows before view_item tracking existed)."""
    fronts = [f for f in shop.fronts if f.ga4_table or _has(shop, "ga4")]
    if not fronts:
        return {}

    def fetch(f, from_pv, segment):
        if not f.ga4_table:
            raise RuntimeError("no ga4_table configured for this shop")
        from google.cloud import bigquery
        project = f.ga4_table.split(".")[0]
        client = bigquery.Client(project=project)
        last = end or dt.date.today() - dt.timedelta(days=1)
        start = last - dt.timedelta(days=days - 1)
        params = [
            bigquery.ScalarQueryParameter("start", "STRING", start.strftime("%Y%m%d")),
            bigquery.ScalarQueryParameter("end", "STRING", last.strftime("%Y%m%d")),
            bigquery.ScalarQueryParameter("max_views", "INT64", MAX_SESSION_VIEWS),
        ]
        sql = GA4_SQL
        if from_pv:
            cat = catalog(shop)["products"]
            host = f.origin.split("://", 1)[1].replace(".", r"\.")
            params += [
                bigquery.ArrayQueryParameter("slugs", "STRING", [p["slug"] for p in cat]),
                bigquery.ArrayQueryParameter("ids", "INT64", [int(p["id"]) for p in cat]),
                bigquery.ScalarQueryParameter("pdp_re", "STRING", rf"^https?://(?:www\.)?{host}/product/([^/?#]+)"),
            ]
            sql = GA4_PV_SQL.replace("ids[OFFSET(o)]", "@ids[OFFSET(o)]")
        seg = f"AND ({segment})" if segment else ""
        job = client.query(sql.format(table=f.ga4_table, segment=seg),
                           job_config=bigquery.QueryJobConfig(query_parameters=params))
        return {"start": str(start), "end": str(last), "rows": [dict(r) for r in job.result()]}

    products, coview, subst = {}, {}, {}
    span = None
    parts = []
    for f in fronts:
        from_pv = pageviews or not f.ga4_view_item
        name = f"ga4_to_{end}.json" if end else "ga4.json"
        if len(shop.fronts) > 1:
            name = f"ga4_{f.key}" + (f"_to_{end}" if end else "") + ".json"
        if from_pv:
            name = name.replace(".json", f"_pv{days}.json")
        segs = list(enumerate(f.ga4_blend)) if f.ga4_blend else [(None, f.ga4_segment)]
        got = []
        for i, seg in segs:
            nm = name if i is None else name.replace(".json", f"_blend{i}.json")
            try:
                got.append(_cached(shop, nm, refresh, lambda seg=seg: fetch(f, from_pv, seg)))
            except Exception as e:  # noqa: BLE001
                print(f"ga4 {f.key}: failed ({e}), ranking without its behaviour", file=sys.stderr)
        # Blended groups count equally: each is scaled to the groups' average product views.
        tot = [sum(r["views"] for r in d["rows"] if r["kind"] == "product") or 1 for d in got]
        parts += [(d, (sum(tot) / len(tot) / t) if f.ga4_blend else 1.0) for d, t in zip(got, tot)]
    for data, w in parts:
        span = span or (data["start"], data["end"])
        for r0 in data["rows"]:
            r = {**r0, "views": r0["views"] * w, "bought": (r0["bought"] or 0) * w} if w != 1.0 else {**r0, "bought": r0["bought"] or 0}
            if r["kind"] == "product":
                g = products.setdefault(r["a"], {"views": 0, "bought": 0, "price": None})
                g["views"] += r["views"]
                g["bought"] += r["bought"]
                g["price"] = g["price"] or r["price"]
            elif r["kind"] == "coview":
                n = coview.get((r["a"], r["b"]), 0) + r["views"]
                coview[(r["a"], r["b"])] = coview[(r["b"], r["a"])] = n
            else:
                subst[(r["a"], r["b"])] = subst.get((r["a"], r["b"]), 0) + r["views"]
    if span is None:
        return {}
    return {"start": span[0], "end": span[1], "products": products, "coview": coview, "subst": subst}


# --- Search demand (GSC export) -------------------------------------------------------------

GSC_SQL = """
SELECT url, SUM(impressions) AS impressions, SUM(clicks) AS clicks
FROM `{table}`
WHERE search_type = 'WEB' AND url LIKE @prefix AND query IS NOT NULL AND NOT is_anonymized_query
  AND NOT REGEXP_CONTAINS(LOWER(query), @brand) {country}
GROUP BY url
"""


def search_demand(shop: Shop, refresh: bool = False) -> dict[str, dict]:
    """Non-brand impressions and clicks per product slug, over the whole export (all fronts)."""
    out: dict[str, dict] = {}
    for f in shop.fronts:
        if not f.gsc_table and not _has(shop, "gsc"):
            continue

        def fetch(f=f):
            if not f.gsc_table:
                raise RuntimeError("no gsc_table configured for this shop")
            from google.cloud import bigquery
            client = bigquery.Client(project=f.gsc_table.split(".")[0])
            country = f"AND country = '{f.gsc_country}'" if f.gsc_country else ""
            job = client.query(GSC_SQL.format(table=f.gsc_table, country=country), job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("prefix", "STRING", f"%/product/%"),
                    bigquery.ScalarQueryParameter("brand", "STRING", f.brand_regex),
                ]))
            return [dict(r) for r in job.result()]

        try:
            rows = _cached(shop, "gsc.json" if len(shop.fronts) == 1 else f"gsc_{f.key}.json", refresh, fetch)
        except Exception as e:  # noqa: BLE001
            print(f"gsc {f.key}: failed ({e}), ranking without its search demand", file=sys.stderr)
            continue
        for r in rows:
            slug = r["url"].split("/product/", 1)[1].split("?")[0].split("#")[0].strip("/")
            agg = out.setdefault(slug, {"impressions": 0, "clicks": 0})
            agg["impressions"] += r["impressions"]
            agg["clicks"] += r["clicks"]
    return out


# --- Link equity (TIPR) ------------------------------------------------------------------------

def link_equity(shop: Shop) -> dict[str, float]:
    """True PR per product slug from the shop's latest TIPR run (absent for shops without one)."""
    path: Path | None = shop.tipr_pages
    if not path or not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["type"] == "product" and r["true_pr"]:
                out[r["url"].split("/product/", 1)[1]] = float(r["true_pr"])
    return out
