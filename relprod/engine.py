"""Pick the 4 related products for every PDP.

1. Relevance(A, B): tag similarity (IDF-weighted subcategory overlap per facet) blended with shopper
   behaviour (sessions that viewed both, or viewed A and bought B instead).
2. Hard filters: B is linkable (published, not test, in season), not a near-duplicate of A or of a
   card already picked, respects exclusive occasions, and clears MIN_RELEVANCE.
3. Four slots, each with its own price band and ranking (config.SLOTS). The three conversion
   slots are filled in order of traffic, so the busiest PDPs get first pick before caps bind.
4. Coverage: the discovery slots are handed first to products related from fewer than
   MIN_INBOUND PDPs (hardest to place first), then filled normally. Products still short take
   the step-up slot of a quiet PDP when they are a genuine step up there.

Relevance is computed for all pairs at once (numpy), and each source keeps the list of candidates
that clear MIN_RELEVANCE and the hard filters, so a 4,000-product shop runs in minutes.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config as C
from .features import Product, idf

COLOUR_WORDS = {w for words in C.COLOURS.values() for w in words} | {"multicolor", "multicolour", "rainbow"}


@dataclass
class Pick:
    source: int
    position: int
    slot: str
    target: int
    relevance: float
    tags: float
    evidence: float           # coview + SUBST_FACTOR * subst sessions
    coview: int
    subst: int
    band_step: int            # 0 = first price band; higher = widened
    reason: str
    repaired: bool = False


class Engine:
    def __init__(self, products: dict[int, Product], beh: dict, feedback: dict | None = None,
                 season_tags: frozenset[str] = frozenset(), season_boost: float = 1.0):
        self.p = products
        # Active seasons (seasons.py): candidates carrying these tags score season_boost higher.
        self.season_tags = season_tags
        self.season_boost = season_boost
        # Learned per card (learn.py): its click rate vs the average card, as a score multiplier.
        self.feedback = {(c["a"], c["b"]): c["factor"] for c in (feedback or {}).get("cards", [])}
        self.idf = idf(products)
        self.coview = beh.get("coview", {}) if beh else {}
        self.subst = beh.get("subst", {}) if beh else {}
        self.p0 = (sum(x.cvr for x in products.values()) / len(products)) if products else 0.02
        self.linkable = [x for x in products.values() if x.linkable]
        self._pct_demand = _percentiles({x.id: x.search_impressions for x in self.linkable})
        self._pct_pr = _percentiles({x.id: x.true_pr for x in self.linkable if x.true_pr is not None})
        self._ix = {pid: i for i, pid in enumerate(products)}
        self._tags = self._tag_matrix()
        self._coll, self._desc = self._collection_matrix(), self._description_matrix()
        self._content = self._content_matrix()
        self._rel = self._content.copy()
        self._blend_behaviour()
        self._cands: dict[int, list[Product]] = {}

    # --- relevance --------------------------------------------------------------------------

    def _tag_matrix(self, sets_of=None, weights=None, idf_of=None) -> np.ndarray:
        """tags(a, b) for every pair: per facet the source has tags on, the IDF-weighted overlap
        share, discounted when the shared tags are common: two products that are both just
        "flowers" or just "birthday" are not as alike as two "new-mom-and-baby" gifts.
        Also used for collections (one facet, its own IDF)."""
        ps = list(self.p.values())
        n = len(ps)
        num = np.zeros((n, n))
        den = np.zeros(n)
        sets_of = sets_of or (lambda x, f: x.facets.get(f, set()))
        idf_of = idf_of or self.idf
        for f, w in (weights or C.FACET_WEIGHTS).items():
            subs = sorted({s for x in ps for s in sets_of(x, f)})
            if not subs:
                continue
            col = {s: j for j, s in enumerate(subs)}
            m = np.zeros((n, len(subs)))
            for i, x in enumerate(ps):
                for s in sets_of(x, f):
                    m[i, col[s]] = 1.0
            weighted = m * np.array([idf_of[s] for s in subs])
            total = weighted.sum(axis=1)
            has = m.any(axis=1)
            for lo in range(0, n, 1024):   # row blocks keep memory flat on big catalogs
                hi = min(n, lo + 1024)
                shared = m[lo:hi] @ weighted.T
                union = total[lo:hi, None] + total[None, :] - shared
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = np.where(union > 0, shared / union, 0.0)
                term = w * share * (1 - np.exp(-shared / C.TAG_SPECIFICITY))
                num[lo:hi] += np.where(has[lo:hi, None], term, 0.0)
            den += np.where(has, w, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den[:, None] > 0, num / den[:, None], 0.0)

    def _collection_matrix(self) -> np.ndarray:
        n = len(self.p)
        df = Counter(c for x in self.p.values() for c in x.collections)
        if not df:
            return np.zeros((n, n))
        cidf = {c: math.log((n + 1) / (k + 1)) + 0.1 for c, k in df.items()}
        return self._tag_matrix(lambda x, f: x.collections, {"collection": 1.0}, cidf)

    def _description_matrix(self) -> np.ndarray:
        """TF-IDF cosine of the products' own text. Words on most products (boilerplate, "bouquet")
        weigh little; shared rare words ("lisianthus", "eucalyptus", "ferrero") weigh a lot."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [x.text for x in self.p.values()]
        n = len(texts)
        if sum(1 for t in texts if t.strip()) < 2:
            return np.zeros((n, n))
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        self._vec = TfidfVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS | C.MARKETING_WORDS), min_df=2, max_df=0.15,
                                    sublinear_tf=True, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-z][a-z]+\b")
        self._tfidf = self._vec.fit_transform(texts)
        sim = (self._tfidf @ self._tfidf.T).toarray()
        np.fill_diagonal(sim, 1.0)
        return sim

    def _content_matrix(self) -> np.ndarray:
        """Weighted blend of subcats, collections and description, renormalised per source over
        the signals it has (a product in no themed collection is judged on the other two)."""
        ps = list(self.p.values())
        has = {"subcats": np.array([bool(x.facets) for x in ps]),
               "collections": np.array([bool(x.collections) for x in ps]),
               "description": np.array([bool(x.text.strip()) for x in ps])}
        mats = {"subcats": self._tags, "collections": self._coll, "description": self._desc}
        num = np.zeros_like(self._tags)
        den = np.zeros(len(ps))
        for k, w in C.CONTENT_WEIGHTS.items():
            num += np.where(has[k][:, None], w * mats[k], 0.0)
            den += np.where(has[k], w, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den[:, None] > 0, num / den[:, None], 0.0)

    def content(self, a: Product, b: Product) -> dict[str, float]:
        ia, ib = self._ix[a.id], self._ix[b.id]
        return {"content": float(self._content[ia, ib]), "subcats": float(self._tags[ia, ib]),
                "collections": float(self._coll[ia, ib]), "description": float(self._desc[ia, ib])}

    def shared_words(self, a: Product, b: Product, k: int = 4) -> list[str]:
        """The description terms that make a and b alike, strongest first."""
        if not hasattr(self, "_tfidf"):
            return []
        va, vb = self._tfidf[self._ix[a.id]], self._tfidf[self._ix[b.id]]
        both = va.multiply(vb)
        if not both.nnz:
            return []
        names = self._vec.get_feature_names_out()
        top = sorted(zip(both.indices, both.data), key=lambda t: -t[1])
        out = []
        for j, _ in top:
            w = names[j]
            if not any(w in o or o in w for o in out) and w not in a.name.lower():
                out.append(w)
            if len(out) == k:
                break
        return out

    def _blend_behaviour(self) -> None:
        """Relevance = tags, blended with behaviour on pairs that have evidence. Behaviour is
        normalised per source by its strongest linkable pair; its weight grows with evidence."""
        raw: dict[int, dict[int, float]] = {}
        for (x, y) in set(self.coview) | set(self.subst):
            a, b = self.p.get(x), self.p.get(y)
            if a is None or b is None or a.id == b.id:
                continue
            e, _, _ = self.evidence(a, b)
            if e:
                raw.setdefault(a.id, {})[b.id] = e / math.sqrt(max(a.views, 1) * max(b.views, 1))
        for aid, row in raw.items():
            bmax = max((v for bid, v in row.items() if self.p[bid].linkable), default=0.0)
            if not bmax:
                continue
            ia = self._ix[aid]
            a = self.p[aid]
            for bid, v in row.items():
                e, _, _ = self.evidence(a, self.p[bid])
                w = C.BEHAVIOUR_WEIGHT * min(1.0, e / C.BEHAVIOUR_FULL_AT)
                ib = self._ix[bid]
                self._rel[ia, ib] = (1 - w) * self._content[ia, ib] + w * v / bmax

    def tags(self, a: Product, b: Product) -> float:
        return float(self._tags[self._ix[a.id], self._ix[b.id]])

    def evidence(self, a: Product, b: Product) -> tuple[float, int, int]:
        cv = self.coview.get((a.id, b.id), 0)
        st = self.subst.get((a.id, b.id), 0)
        return cv + C.SUBST_FACTOR * st, cv, st

    def relevance(self, a: Product, b: Product) -> tuple[float, float, float]:
        """(relevance, content similarity, behaviour evidence)."""
        ia, ib = self._ix[a.id], self._ix[b.id]
        return float(self._rel[ia, ib]), float(self._content[ia, ib]), self.evidence(a, b)[0]

    def candidates(self, a: Product) -> list[Product]:
        """Linkable products relevant enough for a and allowed next to it, in catalog order."""
        key = a.id
        if key not in self._cands:
            ia = self._ix[a.id]
            row, con = self._rel[ia], self._content[ia]
            self._cands[key] = [b for b in self.linkable
                                if row[self._ix[b.id]] >= C.MIN_RELEVANCE and con[self._ix[b.id]] >= C.CONTENT_FLOOR
                                and self.allowed(a, b)]
        return self._cands[key]

    # --- filters ------------------------------------------------------------------------------

    def allowed(self, a: Product, b: Product) -> bool:
        if b.id == a.id or not b.linkable or is_dup(a, b):
            return False
        if a.strict != b.strict:
            return False   # strict lanes (funeral): in only with the same tag, whatever else is tagged
        if a.exclusive_only and not (a.exclusive & b.exclusive):
            return False   # a sympathy shopper sees only sympathy products
        if b.exclusive_only and not (a.exclusive & b.exclusive):
            return False   # a funeral-only product never shows on a birthday bouquet
        if a.lead is not None and not sold_where(a, b, a.lead):
            return False   # the storefront would hide it and show a random product instead
        return True

    def need(self, b: Product, inbound: int) -> float:
        """0..1: how much b needs links. Search demand it is not getting equity for, plus coverage."""
        seo = max(0.0, self._pct_demand.get(b.id, 0) - self._pct_pr.get(b.id, 0.5))
        cover = 1.0 if inbound < C.MIN_INBOUND else 0.0
        return min(1.0, 0.6 * seo + 0.4 * cover)

    # --- slot scores --------------------------------------------------------------------------

    def score(self, rank: str, a: Product, b: Product, rel: float, ev: float, inbound: int) -> float:
        q = math.sqrt(b.cvr / self.p0) if self.p0 else 1.0
        if rank == "closest":
            return rel * (1.25 if looks_different(a, b) else 1.0) * q
        if rank == "proven":
            # Real evidence first; without it this slot behaves like "closest".
            return (1.0 + math.log1p(ev)) * rel * q if ev >= 2 else 0.25 * rel * q
        if rank == "step_up":
            return rel * b.cvr * b.price
        if rank == "discovery":
            return rel * (0.3 + self.need(b, inbound))
        raise ValueError(rank)

    # --- assignment -----------------------------------------------------------------------------

    def run(self, pinned: dict[int, list[int]] | None = None) -> tuple[dict[int, list[Pick]], Counter]:
        """pinned: {source: [card ids in order]} kept exactly as given (e.g. live sets the user
        wants left alone). They count toward inbound; every other PDP is built around them."""
        inbound: Counter = Counter()
        picks: dict[int, list[Pick]] = {}
        self._pinned = set(pinned or {})
        for aid, ids in (pinned or {}).items():
            picks[aid] = []
            for pos, bid in enumerate(ids, 1):
                if aid in self.p and bid in self.p:
                    rel, t, ev = self.relevance(self.p[aid], self.p[bid])
                    _, cv, st = self.evidence(self.p[aid], self.p[bid])
                else:
                    rel = t = ev = 0.0
                    cv = st = 0
                slot = C.SLOTS[min(pos, len(C.SLOTS)) - 1]["id"]
                picks[aid].append(Pick(aid, pos, slot, bid, rel, t, ev, cv, st, 0,
                                       "Pinned: live set kept as is", repaired=True))
                inbound[bid] += 1
        sources = sorted((x for x in self.p.values() if x.id not in self._pinned),
                         key=lambda x: (-x.views, -x.orders, x.id))
        conversion, discovery = C.SLOTS[:-1], C.SLOTS[-1]
        # 1. Conversion slots, busiest PDPs first.
        for a in sources:
            chosen: list[Pick] = []
            for i, slot in enumerate(conversion):
                pk = self._fill(a, slot, i + 1, chosen, inbound)
                if pk:
                    chosen.append(pk)
                    inbound[pk.target] += 1
            picks[a.id] = chosen
        # 2. Discovery slots go first to products still short of MIN_INBOUND links.
        self._cover(picks, inbound, discovery)
        # 3. Remaining discovery slots: normal discovery ranking.
        for a in sources:
            if not any(c.slot == discovery["id"] for c in picks[a.id]):
                pk = self._fill(a, discovery, len(C.SLOTS), picks[a.id], inbound)
                if pk:
                    picks[a.id].append(pk)
                    inbound[pk.target] += 1
        # 4. Still short: take the step-up slot of quiet PDPs (bottom half by traffic).
        self._repair(picks, inbound, sources[len(sources) // 2:])
        # 5. A slot nothing qualified for (e.g. no dearer relevant product) takes the next-best
        #    alternative, so every PDP shows 4 cards.
        for a in sources:
            filled = {c.slot for c in picks[a.id]}
            for i, slot in enumerate(C.SLOTS):
                if slot["id"] not in filled:
                    pk = self._fill(a, {**C.FALLBACK_SLOT, "id": slot["id"]}, i + 1, picks[a.id], inbound)
                    if pk:
                        pk.reason = "Fallback (nothing qualified for this slot): " + pk.reason
                        picks[a.id].append(pk)
                        inbound[pk.target] += 1
        # 6. Every PDP shows 4 cards. Still short (a lone niche item): loosen relevance and price
        #    only. Hubs are a hard rule (a card the shopper cannot buy in their city is pointless),
        #    and so are the strict and exclusive lanes, duplicates and MAX_INBOUND.
        for a in sources:
            if len(picks[a.id]) < C.CARDS:
                self._force_fill(a, picks[a.id], inbound)
        # 7. Products still short of MIN_INBOUND that do have relevant, sellable sources: swap a
        #    non-closest card on one of those sources whose product keeps > MIN_INBOUND. Any price.
        self._rescue(picks, inbound)
        for a in sources:
            picks[a.id].sort(key=lambda c: c.position)
        return picks, inbound

    def _rescue(self, picks: dict[int, list[Pick]], inbound: Counter) -> None:
        sources_of: dict[int, list[Product]] = {}
        for a in self.p.values():
            if a.id in self._pinned:
                continue
            for b in self.candidates(a):
                if inbound[b.id] < C.MIN_INBOUND:
                    sources_of.setdefault(b.id, []).append(a)
        anyprice = (0.0, math.inf)
        for bid in sorted(sources_of, key=lambda k: (len(sources_of[k]), inbound[k], k)):
            b = self.p[bid]
            for a in sorted(sources_of[bid], key=lambda x: -self.relevance(x, b)[0]):
                if inbound[bid] >= C.MIN_INBOUND:
                    break
                row = picks[a.id]
                if any(c.target == bid for c in row):
                    continue
                swaps = [c for c in row if c.slot != "closest" and inbound[c.target] > C.MIN_INBOUND
                         and self._candidate_ok(a, b, [x for x in row if x is not c], anyprice)]
                if not swaps:
                    continue
                cur = max(swaps, key=lambda c: (inbound[c.target], -c.relevance))
                rel, t, ev = self.relevance(a, b)
                _, cv, st = self.evidence(a, b)
                slot = next(s for s in C.SLOTS if s["id"] == cur.slot)
                inbound[cur.target] -= 1
                inbound[bid] += 1
                row[row.index(cur)] = Pick(a.id, cur.position, cur.slot, bid, rel, t, ev, cv, st, cur.band_step,
                                           "Coverage: " + self.reason(a, b, slot, cv, st), repaired=True)

    def _force_fill(self, a: Product, chosen: list[Pick], inbound: Counter) -> None:
        """Top a short row up to C.CARDS with the most relevant allowed products, any price."""
        ia = self._ix[a.id]
        options = sorted((b for b in self.linkable if self.allowed(a, b)),
                         key=lambda b: (-self._rel[ia, self._ix[b.id]], -self._content[ia, self._ix[b.id]], b.id))
        for b in options:
            if len(chosen) >= C.CARDS:
                return
            if any(c.target == b.id for c in chosen) or inbound[b.id] >= C.MAX_INBOUND:
                continue
            if any(is_dup(self.p[c.target], b) for c in chosen) or family_clash(a, [self.p[c.target] for c in chosen], b):
                continue
            used = {c.position for c in chosen}
            pos = next(i for i in range(1, C.CARDS + 1) if i not in used)
            slot = C.SLOTS[pos - 1]
            rel, t, ev = self.relevance(a, b)
            _, cv, st = self.evidence(a, b)
            chosen.append(Pick(a.id, pos, slot["id"], b.id, rel, t, ev, cv, st, len(C.PRICE_WIDEN),
                               "Fallback (minimum 4 cards, any price or relevance): " + self.reason(a, b, slot, cv, st)))
            inbound[b.id] += 1

    def _fill(self, a: Product, slot: dict, position: int, chosen: list[Pick], inbound: Counter,
              exclude: set[int] = frozenset()) -> Pick | None:
        taken = {c.target for c in chosen}
        lo, hi = slot["price"]
        for step, widen in enumerate(C.PRICE_WIDEN):
            options = []
            for b in self.candidates(a):
                if b.id in taken or b.id in exclude or inbound[b.id] >= C.MAX_INBOUND:
                    continue
                if any(is_dup(self.p[c.target], b) for c in chosen):
                    continue
                if family_clash(a, [self.p[c.target] for c in chosen], b):
                    continue
                ratio = b.price / a.price if a.price else 1.0
                if widen is not None and not (lo / widen <= ratio <= hi * widen):
                    continue
                if slot.get("min_ratio") and ratio < slot["min_ratio"]:
                    continue  # a step up is never cheaper, however far the band widens
                rel, t, ev = self.relevance(a, b)
                if rel < C.MIN_RELEVANCE:
                    continue
                options.append((b, rel, t, ev))
            if options and slot.get("similar_within"):
                top = max(o[2] for o in options)
                options = [o for o in options if o[2] >= slot["similar_within"] * top]
            best = None
            for b, rel, t, ev in options:
                s = self.score(slot["rank"], a, b, rel, ev, inbound[b.id])
                # Variety across the 4 cards: a second card with the same look scores lower.
                if any(same_look(self.p[c.target], b) for c in chosen):
                    s *= 0.8
                s *= self.feedback.get((a.id, b.id), 1.0)
                if self.season_tags and b.subs & self.season_tags:
                    s *= self.season_boost
                # Shared table (several fronts): a card missing on another front the product is on gets
                # replaced there by a random product, so prefer cards that work on every front.
                if len(a.fronts) > 1 and not all(sold_where(a, b, f) for f in a.fronts):
                    s *= 0.9
                if best is None or s > best[0]:
                    best = (s, b, rel, t, ev)
            if best:
                _, b, rel, t, ev = best
                _, cv, st = self.evidence(a, b)
                return Pick(a.id, position, slot["id"], b.id, rel, t, ev, cv, st, step,
                            self.reason(a, b, slot, cv, st))
        return None

    def _candidate_ok(self, a: Product, b: Product, chosen: list[Pick], band: tuple[float, float]) -> bool:
        ia, ib = self._ix[a.id], self._ix[b.id]
        if any(c.target == b.id for c in chosen) or self._rel[ia, ib] < C.MIN_RELEVANCE:
            return False
        if self._content[ia, ib] < C.CONTENT_FLOOR:
            return False
        if not self.allowed(a, b):
            return False
        if any(is_dup(self.p[c.target], b) for c in chosen):
            return False
        if family_clash(a, [self.p[c.target] for c in chosen], b):
            return False
        ratio = b.price / a.price if a.price else 1.0
        return band[0] <= ratio <= band[1] and self.relevance(a, b)[0] >= C.MIN_RELEVANCE

    def _cover(self, picks: dict[int, list[Pick]], inbound: Counter, slot: dict) -> None:
        """Assign empty discovery slots to under-linked products, hardest to place first.

        A product with few possible sources (narrow occasion, unusual price) is placed before one
        that could go almost anywhere. Each takes its most relevant free source, quieter PDPs
        preferred so the busiest pages keep discovery for search demand.
        """
        lo, hi = slot["price"]
        band = (lo / C.PRICE_WIDEN[1], hi * C.PRICE_WIDEN[1])
        busiest = math.log1p(max((x.views for x in self.p.values()), default=1) or 1)
        position = len(C.SLOTS)
        # Placing b on a only takes a's slot, so options are built once and then shrink: a leaves
        # every list, b leaves when it reaches MIN_INBOUND. Same result as rebuilding each round.
        free = {aid for aid, ch in picks.items() if aid not in self._pinned
                and not any(c.slot == slot["id"] for c in ch)}
        sources_of: dict[int, list[int]] = {}
        for aid in picks:
            if aid in free:
                for b in self.candidates(self.p[aid]):
                    if inbound[b.id] < C.MIN_INBOUND:
                        sources_of.setdefault(b.id, []).append(aid)
        options = {bid: [aid for aid in aids if self._candidate_ok(self.p[aid], self.p[bid], picks[aid], band)]
                   for bid, aids in sources_of.items()}
        options = {k: v for k, v in options.items() if v}
        while options:
            bid = min(options, key=lambda k: (len(options[k]), inbound[k], k))
            b = self.p[bid]
            aid = max(options[bid], key=lambda x: self.relevance(self.p[x], b)[0]
                      - 0.3 * math.log1p(self.p[x].views) / busiest)
            a = self.p[aid]
            rel, t, ev = self.relevance(a, b)
            _, cv, st = self.evidence(a, b)
            picks[aid].append(Pick(aid, position, slot["id"], bid, rel, t, ev, cv, st, 0,
                                   "Coverage: " + self.reason(a, b, slot, cv, st), repaired=True))
            inbound[bid] += 1
            free.discard(aid)
            for k in list(options):
                if aid in options[k]:
                    options[k].remove(aid)
                if not options[k] or inbound[k] >= C.MIN_INBOUND:
                    del options[k]

    def _repair(self, picks: dict[int, list[Pick]], inbound: Counter, quiet: list[Product]) -> None:
        """Last resort for products still short: replace the step-up card on a quiet PDP, if the
        product is itself a step up (>= 10% dearer) and the card it replaces keeps > MIN_INBOUND."""
        slot = next(s for s in C.SLOTS if s["id"] == "step_up")
        band = (1.10, slot["price"][1] * C.PRICE_WIDEN[1])
        for b in sorted((x for x in self.linkable if inbound[x.id] < C.MIN_INBOUND), key=lambda x: inbound[x.id]):
            options = []
            for a in quiet:
                chosen = picks[a.id]
                cur = next((c for c in chosen if c.slot == "step_up" and not c.repaired), None)
                if not cur or inbound[cur.target] <= C.MIN_INBOUND:
                    continue
                others = [c for c in chosen if c is not cur]
                if self._candidate_ok(a, b, others, band):
                    options.append((self.relevance(a, b)[0], a, cur))
            for _, a, cur in sorted(options, key=lambda o: -o[0]):
                if inbound[b.id] >= C.MIN_INBOUND:
                    break
                if inbound[cur.target] <= C.MIN_INBOUND:
                    continue
                rel, t, ev = self.relevance(a, b)
                _, cv, st = self.evidence(a, b)
                inbound[cur.target] -= 1
                inbound[b.id] += 1
                picks[a.id][picks[a.id].index(cur)] = Pick(
                    a.id, cur.position, "step_up", b.id, rel, t, ev, cv, st, 0,
                    "Coverage: " + self.reason(a, b, slot, cv, st), repaired=True)

    # --- explanation -------------------------------------------------------------------------------

    def reason(self, a: Product, b: Product, slot: dict, cv: int, st: int) -> str:
        c = self.content(a, b)
        bits = [f"similarity {c['content']:.2f}"]
        words = self.shared_words(a, b)
        if words:
            bits.append(f"description {c['description']:.2f} ({', '.join(words)})")
        shared = [s for f in ("kind", "flower", "occasion", "recipient")
                  for s in sorted(a.facets.get(f, set()) & b.facets.get(f, set()), key=lambda s: -self.idf[s])]
        if shared:
            bits.append(f"subcats {c['subcats']:.2f} ({', '.join(shared[:3])})")
        colls = sorted(a.collections & b.collections, key=lambda x: len(x))
        if colls:
            bits.append(f"collections {c['collections']:.2f} ({', '.join(colls[:3])})")
        if st:
            bits.append(f"{st} shoppers viewed this and bought it instead")
        if cv:
            bits.append(f"{cv} sessions viewed both")
        if a.price:
            d = (b.price / a.price - 1) * 100
            bits.append(f"price {b.price:.0f} vs {a.price:.0f} ({d:+.0f}%)")
        if slot["id"] == "closest" and looks_different(a, b):
            bits.append("different look")
        if slot["id"] == "discovery":
            if b.search_impressions >= 20:
                bits.append(f"{b.search_impressions:,} non-brand impressions")
        return "; ".join(bits)


# --- helpers -------------------------------------------------------------------------------------

def sold_where(a: Product, b: Product, front: str) -> bool:
    """b is on `front` and on sale wherever a is: in every hub a is sold in there, and, when b's
    vendors deliver only to some cities, in every city a is limited to (a product sold in every
    city needs cards sold in every city). The storefront hides a card the shopper's hub or city
    does not sell and shows a random product instead. Hard rule."""
    hubs = b.fronts.get(front)
    if hubs is None:
        return False
    mine = a.fronts.get(front)
    if mine and not mine <= hubs:
        return False
    if not mine:
        hub = a.need_hub.get(front)
        if hub is not None and hub not in hubs:
            return False
    if front in b.cities:
        return front in a.cities and a.cities[front] <= b.cities[front]
    return True


def is_dup(a: Product, b: Product) -> bool:
    if not a.name_tokens or not b.name_tokens:
        return False
    inter = len(a.name_tokens & b.name_tokens)
    if inter == min(len(a.name_tokens), len(b.name_tokens)):
        return True   # one name contains the other ("Sunny Wishes" / "Sunny Wishes Bundle")
    return inter / len(a.name_tokens | b.name_tokens) >= C.NAME_DUP_JACCARD


def same_family(a: Product, b: Product) -> bool:
    """Colour variants of one product ("Pink Tulips in Vase" / "Yellow Tulips in Vase")."""
    ta, tb = a.name_tokens - COLOUR_WORDS, b.name_tokens - COLOUR_WORDS
    if not ta or not tb:
        return False
    if min(len(ta), len(tb)) < 2:
        return ta == tb   # "Blue and White Wreath" is not a variant of every other wreath
    inter = len(ta & tb)
    return inter == min(len(ta), len(tb)) or inter / len(ta | tb) >= C.NAME_DUP_JACCARD


def family_clash(a: Product, chosen: list[Product], b: Product) -> bool:
    """At most one colour variant of the source per PDP, and no two cards from one family."""
    if any(same_family(c, b) for c in chosen):
        return True
    return same_family(a, b) and any(same_family(a, c) for c in chosen)


def looks_different(a: Product, b: Product) -> bool:
    fa, fb = a.facets.get("flower", set()), b.facets.get("flower", set())
    return (bool(a.colours) and bool(b.colours) and not (a.colours & b.colours)) or (bool(fa) and bool(fb) and not (fa & fb))


def same_look(a: Product, b: Product) -> bool:
    return bool(a.colours) and a.colours == b.colours and a.facets.get("flower") == b.facets.get("flower")


def _percentiles(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    order = sorted(values, key=lambda k: values[k])
    n = len(order)
    return {k: i / (n - 1) if n > 1 else 0.5 for i, k in enumerate(order)}
