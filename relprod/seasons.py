"""Seasonal calendar: when each shop's sets should change, and how.

Measure your own peaks first (daily orders, and the share of orders that are seasonal products,
over two years). On the gifting shops this was built for, the pattern was:

  Valentine's    10-18x normal daily orders, ramp from ~10-20 days before, over ~4 days after
  Mother's Day   3-12x, ramp ~10-14 days; most of that week's orders are Mother's Day products
  Christmas      volume only ~2-3x, but Christmas products go from ~1-9% to 10-25% of orders
  Sympathy days  funeral products several times their normal share in the week before

Two levers while a season is on (run.py build --date picks the active seasons):
  1. Tag boost: candidates carrying the season's tags score SEASON_BOOST higher, so seasonal
     products take cards on related PDPs. Only for seasons whose tags are specific (Christmas,
     sympathy); Valentine's and Mother's Day tags often sit on half the catalog.
  2. Last year's behaviour: what shoppers viewed together and bought instead in the same window
     last year is added to the 12-month behaviour (weight LAST_YEAR_WEIGHT), so the peak's own
     pairs (e.g. Valentine's bestsellers) rise. This is the lever for the volume peaks.

`on` leaves 3 days for review, upload and the storefront cache before the ramp starts; `off` is
when demand is back to normal.
"""

import datetime as dt
from dataclasses import dataclass

SEASON_BOOST = 1.5
LAST_YEAR_WEIGHT = 2.0


@dataclass(frozen=True)
class Season:
    key: str
    name: str
    on: dt.date
    off: dt.date
    event: dt.date
    tags: frozenset[str] = frozenset()      # subcategory slugs (after sub_aliases) to boost
    last_year: tuple[dt.date, dt.date] | None = None   # same window last year, for behaviour
    why: str = ""

    def active(self, day: dt.date) -> bool:
        return self.on <= day <= self.off


D = dt.date


def _second_sunday_of_may(year: int) -> dt.date:
    first = D(year, 5, 1)
    return first + dt.timedelta(days=(6 - first.weekday()) % 7 + 7)


def _yearly(key, name, event_of, before, after, tags=frozenset(), why=""):
    """The season for this year and next, each with last year's window for behaviour."""
    out = []
    for year in range(dt.date.today().year, dt.date.today().year + 2):
        ev, last = event_of(year), event_of(year - 1)
        out.append(Season(key, name, ev - dt.timedelta(days=before), ev + dt.timedelta(days=after), ev, frozenset(tags),
                          (last - dt.timedelta(days=before), last + dt.timedelta(days=after)), why))
    return out


SEASONS: dict[str, list[Season]] = {
    "demo": [
        *_yearly("valentines", "Valentine's", lambda y: D(y, 2, 14), 17, 4, (),
                 "volume peak: last year's behaviour only (the tag sits on too many products)"),
        *_yearly("mothers-day", "Mother's Day", _second_sunday_of_may, 14, 4, ("mothers-day",),
                 "volume peak plus a specific tag"),
        *_yearly("christmas", "Christmas", lambda y: D(y, 12, 25), 21, 1, ("christmas",),
                 "share peak: Christmas products take cards on related PDPs"),
    ],
}


def active(shop_key: str, day: dt.date) -> list[Season]:
    return [s for s in SEASONS.get(shop_key, []) if s.active(day)]


def calendar(shop_key: str) -> list[dict]:
    """Switch dates: rebuild + upload on `on`, rebuild without the season on `off` + 1."""
    out = []
    for s in sorted(SEASONS.get(shop_key, []), key=lambda s: s.on):
        lever = ("tag boost + last year's behaviour" if s.tags else "last year's behaviour")
        out.append({"shop": shop_key, "season": s.name, "event": str(s.event), "switch_on": str(s.on),
                    "switch_off": str(s.off + dt.timedelta(days=1)), "lever": lever, "evidence": s.why})
    return out


if __name__ == "__main__":
    import csv
    import sys
    rows = [r for k in SEASONS for r in calendar(k)]
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
