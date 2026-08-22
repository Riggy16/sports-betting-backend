from __future__ import annotations

import pandas as pd

from .config import ModelConfig, DEFAULT_CONFIG


def validate_split_config(config: ModelConfig = DEFAULT_CONFIG) -> None:
    train = set(config.training_seasons)
    val = set(config.validation_seasons)
    hold = set(config.holdout_seasons)
    if train & val or train & hold or val & hold:
        raise ValueError("Season splits overlap; holdout integrity is broken.")


def threshold_selection_seasons(config: ModelConfig = DEFAULT_CONFIG) -> tuple[int, ...]:
    validate_split_config(config)
    return tuple(config.validation_seasons)


def assign_split(season: int, config: ModelConfig = DEFAULT_CONFIG) -> str:
    validate_split_config(config)
    if season in config.training_seasons:
        return "train"
    if season in config.validation_seasons:
        return "validation"
    if season in config.holdout_seasons:
        return "holdout"
    return "unused"


def split_frame(df: pd.DataFrame, config: ModelConfig = DEFAULT_CONFIG, season_col: str = "season") -> dict[str, pd.DataFrame]:
    validate_split_config(config)
    if season_col not in df.columns:
        raise ValueError(f"Missing required season column: {season_col}")
    labeled = df.copy()
    labeled["_split"] = labeled[season_col].map(lambda s: assign_split(int(s), config))
    return {name: labeled[labeled["_split"] == name].drop(columns=["_split"]) for name in ("train", "validation", "holdout")}
