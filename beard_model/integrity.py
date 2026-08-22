from __future__ import annotations

from collections.abc import Iterable

PROHIBITED_MARKET_TOKENS = (
    "spread",
    "odds",
    "moneyline",
    "market",
    "sportsbook",
    "vig",
    "juice",
    "closing_line",
    "close_line",
    "open_line",
    "opening_line",
    "clv",
)


def market_derived_columns(columns: Iterable[str]) -> list[str]:
    bad: list[str] = []
    for column in columns:
        name = str(column).lower().strip()
        if any(token in name for token in PROHIBITED_MARKET_TOKENS):
            bad.append(str(column))
    return bad


def assert_market_blind_features(columns: Iterable[str]) -> None:
    bad = market_derived_columns(columns)
    if bad:
        raise ValueError(
            "Market-derived features are forbidden in the BEARD fair-line engine: "
            + ", ".join(sorted(bad))
        )
