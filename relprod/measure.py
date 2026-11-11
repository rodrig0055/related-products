"""What shoppers did with the related-products row, from the GA4 export alone (no extra tracking).

On a PDP the only links to other products are the related cards (checked on the TIPR crawl:
every product-to-product link on 339 PDPs sits in the related module), and GA4 records the
previous page as page_referrer on in-site navigation (it matches the previous page_view 97% of
the time). So a "related click" = a product page_view whose referrer is another product page and
whose product is new in the session (a Back to an earlier product is not a click).
/product/search (internal search results) shares the path prefix and is excluded on both ends.

flow() returns, for a date window:
  views    {slug: sessions that viewed the PDP}
  clicks   [{s, src, dst, t}]           one per session x source x destination
  buys     {s: [(t, product_id, revenue)]}  purchases in sessions that had a related click
  sessions [{s, first, pdps, clicked, bought, revenue}]  every session that viewed a PDP
"""

import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict

from .shops import Shop
from .sources import CACHE

FLOW_SQL = """
WITH ev AS (
  SELECT CONCAT(user_pseudo_id, '.', CAST((SELECT value.int_value FROM UNNEST(event_params)
                WHERE key = 'ga_session_id') AS STRING)) AS s,
    event_timestamp AS t, event_name,
    REGEXP_EXTRACT((SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location'), @pdp_re) AS pdp,
    REGEXP_EXTRACT((SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_referrer'), @pdp_re) AS ref_pdp,
    ecommerce.purchase_revenue AS rev, items
  FROM `{table}`
  WHERE _TABLE_SUFFIX BETWEEN @start AND @end AND event_name IN ('page_view', 'purchase') {segment}
),
-- /product/search is the internal search results page, not a product.
pv AS (SELECT s, t, pdp, NULLIF(ref_pdp, 'search') AS ref_pdp FROM ev
       WHERE event_name = 'page_view' AND pdp IS NOT NULL AND pdp != 'search' AND s IS NOT NULL),
first_seen AS (SELECT s, pdp, MIN(t) AS t0 FROM pv GROUP BY s, pdp),
clicks AS (
  SELECT pv.s, pv.ref_pdp AS src, pv.pdp AS dst, MIN(pv.t) AS t
  FROM pv JOIN first_seen f ON f.s = pv.s AND f.pdp = pv.pdp AND f.t0 = pv.t
  WHERE pv.ref_pdp IS NOT NULL AND pv.ref_pdp != pv.pdp
  GROUP BY 1, 2, 3
),
purchases AS (SELECT s, t, rev, items FROM ev WHERE event_name = 'purchase' AND s IS NOT NULL),
sess AS (
  SELECT f.s, ARRAY_AGG(f.pdp ORDER BY f.t0 LIMIT 1)[OFFSET(0)] AS first, COUNT(*) AS pdps
  FROM first_seen f GROUP BY f.s
),
click_n AS (SELECT s, COUNT(*) AS n FROM clicks GROUP BY s),
buy_n AS (SELECT s, COUNT(*) AS n, SUM(rev) AS rev FROM purchases GROUP BY s)
SELECT 'session' AS kind, sess.s, sess.first AS a, CAST(NULL AS STRING) AS b, sess.pdps AS n,
  IFNULL(click_n.n, 0) AS clicked, IFNULL(buy_n.n, 0) AS bought, buy_n.rev AS revenue,
  CAST(NULL AS INT64) AS t, CAST(NULL AS INT64) AS item
FROM sess LEFT JOIN click_n USING (s) LEFT JOIN buy_n USING (s)
UNION ALL
SELECT 'view', CAST(NULL AS STRING), pdp, NULL, COUNT(*), NULL, NULL, NULL, NULL, NULL FROM first_seen GROUP BY pdp
UNION ALL
SELECT 'click', s, src, dst, NULL, NULL, NULL, NULL, t, NULL FROM clicks
UNION ALL
SELECT 'buy', p.s, NULL, NULL, NULL, NULL, NULL, i.price * COALESCE(i.quantity, 1), p.t, SAFE_CAST(i.item_id AS INT64)
FROM purchases p, UNNEST(p.items) i WHERE p.s IN (SELECT s FROM clicks)
"""


def flow(shop: Shop, start: dt.date, end: dt.date, refresh: bool = False, segment: bool = True) -> dict:
    """`shop` is a storefront (Front). Its GA4 segment applies unless segment=False (all visitors)."""
    seg = getattr(shop, "ga4_segment", None) if segment else None
    tag = "" if not seg else "_seg" + hashlib.md5(seg.encode()).hexdigest()[:6]   # one cache per segment
    path = CACHE / shop.key / f"flow{tag}_{start}_{end}.json"
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
    else:
        from google.cloud import bigquery
        host = shop.origin.split("://", 1)[1].replace(".", r"\.")
        client = bigquery.Client(project=shop.ga4_table.split(".")[0])
        job = client.query(FLOW_SQL.format(table=shop.ga4_table, segment=f"AND ({seg})" if seg else ""),
                           job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "STRING", start.strftime("%Y%m%d")),
                bigquery.ScalarQueryParameter("end", "STRING", end.strftime("%Y%m%d")),
                bigquery.ScalarQueryParameter("pdp_re", "STRING", rf"^https?://(?:www\.)?{host}/product/([^/?#]+)"),
            ]))
        rows = [dict(r) for r in job.result()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, default=str))
        print(f"flow {start}..{end}: {len(rows)} rows from GA4", file=sys.stderr)
    out = {"start": str(start), "end": str(end), "days": (end - start).days + 1,
           "views": {}, "clicks": [], "buys": defaultdict(list), "sessions": []}
    for r in rows:
        k = r["kind"]
        if k == "view":
            out["views"][r["a"]] = r["n"]
        elif k == "click":
            out["clicks"].append({"s": r["s"], "src": r["a"], "dst": r["b"], "t": int(r["t"])})
        elif k == "buy":
            out["buys"][r["s"]].append((int(r["t"]), r["item"], float(r["revenue"] or 0)))
        else:
            out["sessions"].append({"s": r["s"], "first": r["a"], "pdps": r["n"], "clicked": r["clicked"],
                                    "bought": r["bought"] > 0, "revenue": float(r["revenue"] or 0)})
    return out


def click_purchases(fl: dict, slug_to_id: dict[str, int]) -> list[dict]:
    """Each related click, with the revenue of the clicked product if it was bought afterwards
    in the same session (the click's direct sale)."""
    out = []
    for c in fl["clicks"]:
        dst_id = slug_to_id.get(c["dst"])
        rev = sum(r for t, item, r in fl["buys"].get(c["s"], []) if item == dst_id and t >= c["t"])
        out.append({**c, "bought": rev > 0, "revenue": rev})
    return out
