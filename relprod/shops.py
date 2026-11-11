"""Per-shop settings. Everything that differs between storefronts lives here; the engine is shared.

To add a shop: copy DEMO, point it at your exported files in cache/<key>/ and check each field:
- `facets`: map each of your category slugs to the facet it describes (occasion, recipient,
  flower, kind). Subcategories inherit their category's facet unless `facet_overrides` says else.
- `exclusive` / `strict`: occasions with their own shopper (sympathy, grand openings). Strict
  lanes link only within themselves.
- `hub_ids`: a single-hub shop names its hub. Multi-hub shops leave it None: every card must be
  sold in the hub the source product mostly sells in (features.need_hub), because a storefront
  that hides out-of-hub cards would otherwise show a gap.
- `fronts_`: several storefronts sharing one related-products table (two brands on one catalog).
  Cards are required on the first front a product is published on and preferred on the others.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Front:
    """One storefront reading the shop's product_related table."""
    key: str
    name: str
    origin: str
    published_col: str
    ga4_table: str | None = None            # GA4 export; None = no export
    ga4_view_item: bool = True              # False: the storefront sends no view_item, use page views
    gsc_table: str | None = None
    brand_regex: str = r"acme"
    currency: str = ""
    # False while this storefront does not read the fixed sets yet (measurement waits for it).
    curated_live: bool = True
    # Only these shoppers' behaviour counts (an SQL predicate on a GA4 event row), and only search
    # demand from this GSC country (e.g. local shoppers, not senders abroad).
    ga4_segment: str | None = None
    gsc_country: str | None = None
    # Shopper groups that count equally in the behaviour signals, whatever their size (SQL
    # predicates), e.g. two device languages weighted 50/50 so the bigger one does not drown the other.
    ga4_blend: tuple[str, ...] = ()


@dataclass(frozen=True)
class Shop:
    key: str
    name: str
    origin: str
    currency: str
    published_col: str = "published"
    product_type_ids: tuple[int, ...] = ()  # giftable products; add-ons are a separate checkout step
    facets: dict[str, str] = field(default_factory=dict)  # category slug -> occasion | recipient | flower | kind
    facet_overrides: dict[str, str] = field(default_factory=dict)  # subcategory slug -> facet
    # Occasions with their own shopper: a source in one sees only products in it, and products
    # tagged ONLY with it never show elsewhere (no funeral wreath on a birthday bouquet).
    exclusive: tuple[str, ...] = ()
    # Stricter: ANY product carrying one of these tags is in that lane, even if it also has other
    # occasions, and it links only to and from products that carry the same tag. Mixed tagging is
    # common (a white tulip bouquet tagged anniversary + graduation + funeral), and one wreath next
    # to a bridal bouquet is worse than a sympathy product missing from a birthday page.
    strict: tuple[str, ...] = ()
    # Occasions that come and go. A product whose occasions are all seasonal is linked only while
    # one of its seasonal subcategories is published.
    seasonal: tuple[str, ...] = ()
    hub_ids: tuple[int, ...] | None = None
    ga4_table: str | None = None            # BigQuery GA4 export, events_* wildcard
    gsc_table: str | None = None            # BigQuery GSC bulk export, searchdata_url_impression
    brand_regex: str = r"acme"
    tipr_pages: Path | None = None          # TIPR pages.csv, for link-equity need
    image_base: str = ""
    # Product ids that must never be linked (tests, internal SKUs).
    blocklist: tuple[int, ...] = ()
    # Subcategory slug -> canonical slug, where a shop tags one idea twice (birthday / birthdays).
    sub_aliases: dict[str, str] = field(default_factory=dict)
    fronts_: tuple[Front, ...] = ()
    # Hubs in order of size, to pick a product's hub when it has no orders.
    hub_order: tuple[int, ...] = ()
    admin_origin: str | None = None         # where the related-products admin screen lives
    ga4_view_item: bool = True              # False: read product views from page_view URLs
    ga4_segment: str | None = None          # see Front
    gsc_country: str | None = None
    ga4_blend: tuple[str, ...] = ()

    @property
    def fronts(self) -> tuple[Front, ...]:
        """Storefronts sharing this shop's table; one implicit front for single-brand shops."""
        return self.fronts_ or (Front(self.key, self.name, self.origin, self.published_col, self.ga4_table,
                                      self.ga4_view_item, self.gsc_table, self.brand_regex, self.currency,
                                      ga4_segment=self.ga4_segment, gsc_country=self.gsc_country,
                                      ga4_blend=self.ga4_blend),)


SHOPS = {
    # The demo shop: a fictional florist and gift shop. Generate its inputs with
    # `python examples/make_demo.py`, then `python run.py build --shop demo`.
    "demo": Shop(
        key="demo",
        name="Acme Florals",
        origin="https://acmeflorals.example",
        currency="USD",
        product_type_ids=(1,),
        facets={
            "occasions": "occasion",
            "recipients": "recipient",
            "flowers": "flower",
            "gifts": "kind",
        },
        exclusive=("sympathy", "grand-opening"),
        strict=("sympathy",),
        seasonal=("valentines", "mothers-day", "christmas"),
        hub_ids=(1,),
        # Point these at your BigQuery exports to pull behaviour and search demand with --refresh:
        # ga4_table="my-project.analytics_123456789.events_*",
        # gsc_table="my-project.searchconsole.searchdata_url_impression",
        brand_regex=r"acme",
        image_base="https://images.unsplash.com/",
    ),
}
