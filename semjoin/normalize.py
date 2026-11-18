"""Value normalization shared by baselines and the hard-pair metrics."""

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    v = unicodedata.normalize("NFKD", value)
    v = "".join(c for c in v if not unicodedata.combining(c))
    v = _PUNCT.sub(" ", v.casefold())
    return _SPACE.sub(" ", v).strip()
