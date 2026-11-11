"""Engine knobs, shared by every shop. Shop-specific settings live in shops.py."""

CARDS = 4  # the PDP shows a fixed set of 4 related products

# How much each facet counts toward "same kind of gift" (tag relevance). Within a facet, rarer
# subcategories count more (IDF), so sharing "funeral" says more than sharing "birthday".
FACET_WEIGHTS = {"occasion": 0.40, "recipient": 0.15, "kind": 0.30, "flower": 0.15}
# Specificity discount: shared tags with total IDF x count (1 - exp(-x / TAG_SPECIFICITY)).
# A tag on most products (IDF ~0.25) keeps ~20%; a rare one (IDF ~3) keeps ~95%.
TAG_SPECIFICITY = 1.0
# Content similarity ("is this the same kind of product?") = weighted blend of three signals, each
# 0..1, renormalised over the ones the source product has:
#   subcats      IDF-weighted subcategory overlap per facet (FACET_WEIGHTS below)
#   collections  IDF-weighted overlap of themed collections (COLLECTION_SKIP drops sales, delivery
#                areas and bestseller lists, which say nothing about the product)
#   description  TF-IDF cosine of the product's own text: name, opening paragraph and the
#                composition sections (Blooms, Inclusions, Components...), not dimensions or notes
CONTENT_WEIGHTS = {"subcats": 0.40, "collections": 0.20, "description": 0.40}
COLLECTION_SKIP = (r"sale|deal|payday|mega|grand|cyber|promo|discount|voucher|\d+-\d+|^\d+$|best-?sell|bestseller|"
                   r"popular|trending|new-arrival|top-|flower-delivery|flower-shop|delivery|nationwide|express|same-day|"
                   r"under-|below-|affordable|budget|surprise-gift-service|all-|^test|oktoberfest|pride$")
DESCRIPTION_KEEP = ("blooms", "inclusions", "inclusion", "components", "ingredients", "material", "what's inside",
                    "contents", "flowers")
# Copywriting words that say nothing about what the product is.
MARKETING_WORDS = frozenset("""life moments moment meaningful thoughtful vibrant perfect gift gifts gifting occasion occasions
special someone love loved loving beautiful beauty charm charming elegant elegance warmth warm heartfelt celebrate
celebration features featuring includes include including adds add lasting impression presentation modern bright fresh
soft tones simply surprise gesture ideal way make makes express feel feels spirit joy joyful delightful refined truly
dream lovely wonderful sweet sweetest touch personal message checkout step craft day days just because perfect
arrangement arranged bouquet blooms bloom piece pieces pcs pc set approximately inches cm size note please""".split())
# Every card must look like the source on content, whatever the behaviour says: two products
# viewed together a lot (two bestsellers) are not "related" if they are different kinds of gift.
CONTENT_FLOOR = 0.15
# Relevance = blend of tags and shopper behaviour. Behaviour gets this share once a pair has
# BEHAVIOUR_FULL_AT sessions of evidence, less below that (tags carry sparse pairs).
BEHAVIOUR_WEIGHT = 0.5
BEHAVIOUR_FULL_AT = 10
SUBST_FACTOR = 3.0      # one "viewed A, bought B instead" session is worth 3 co-view sessions
MIN_RELEVANCE = 0.20    # below this a candidate is off-topic for the source
# Conversion rate is smoothed toward the shop mean with this many pseudo-sessions.
CVR_PRIOR_SESSIONS = 60

# Each slot answers a different reason a shopper does not buy the product in front of them.
# price = allowed price ratio (candidate / source); the band widens step by step if empty.
SLOTS = [
    {"id": "closest", "label": "Closest alternative",
     "why": "Same occasion and price, a different look", "price": (0.80, 1.25), "rank": "closest",
     "similar_within": 0.8},
    {"id": "proven", "label": "Shoppers chose instead",
     "why": "What shoppers who viewed this product bought or viewed instead", "price": (0.60, 1.60),
     "rank": "proven"},
    {"id": "step_up", "label": "Step up",
     "why": "Same occasion, a more generous gift", "price": (1.20, 1.90), "rank": "step_up", "min_ratio": 1.05,
     "similar_within": 0.6},
    {"id": "discovery", "label": "Discovery",
     "why": "Relevant product that needs internal links (search demand, little link equity) "
            "or has too few related links pointing to it", "price": (0.60, 1.30), "rank": "discovery"},
]
# similar_within: a card must be at least this share as similar (content) as the most similar
# candidate in its price band, so conversion rate only chooses among genuinely close products.
PRICE_WIDEN = [1.0, 1.25, 1.6, None]  # band multipliers tried in turn; None = any price
# When a slot finds nothing at any price, it takes the best remaining "closest" candidate.
FALLBACK_SLOT = {"label": "Fallback", "why": "Next-best alternative", "price": (0.5, 2.0), "rank": "closest"}

# Link-graph constraints, across the whole shop.
# Sweep on a ~350-product shop (1,400 cards): min 2 / max 12 links every product at 98.8% of the
# unconstrained conversion value; min 3 was not reachable without off-topic cards.
MIN_INBOUND = 2     # every linkable product is related from at least this many PDPs
MAX_INBOUND = 12    # and from at most this many (best-sellers already top the listing grids)

# "Different look" = differs from the source in colour or in flower type.
COLOURS = {
    "red": ["red", "crimson", "scarlet", "ruby"],
    "pink": ["pink", "blush", "rose gold", "fuchsia", "magenta"],
    "white": ["white", "ivory", "cream", "snow", "pure"],
    "yellow": ["yellow", "sunflower", "sunny", "golden", "gold", "lemon"],
    "orange": ["orange", "peach", "coral", "apricot"],
    "purple": ["purple", "lavender", "lilac", "violet", "mauve"],
    "blue": ["blue", "navy"],
    "green": ["green", "plant", "succulent", "orchid"],
}
# Two products are near-duplicates (never related) when their names share this much.
NAME_DUP_JACCARD = 0.6
NAME_STOPWORDS = {"the", "of", "and", "a", "in", "with", "bouquet", "flowers", "flower", "box", "bundle",
                  "gift", "set", "basket", "hamper", "&", "-", "x"}
