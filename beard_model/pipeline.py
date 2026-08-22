from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import DEFAULT_CONFIG, ModelConfig
from .data import load_pbp, load_schedules
from .features import (
    add_pregame_rolling_features,
    combine_team_game_metrics,
    schedule_team_game_metrics,
    team_game_metrics,
)
from .matchup import build_matchup_frame
from .qb import build_pregame_qb_features, qb_game_metrics


def _attach_qb_features(team_pregame: pd.DataFrame, qb_pregame: pd.DataFrame) -> pd.DataFrame:
    """Attach market-blind QB snapshots to team pregame rows."""
    qb_cols = [
        c for c in qb_pregame.columns
        if c.startswith("pre_qb_") or c in {"game_id", "team", "availability_confidence"}
    ]
    keep = qb_pregame[qb_cols].drop_duplicates(["game_id", "team"])
    return team_pregame.merge(keep, on=["game_id", "team"], how="left", validate="one_to_one")


def build_feature_frame_from_frames(
    schedules: pd.DataFrame,
    pbp_by_season: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    """Create one strictly pregame, market-blind matchup row per scheduled game."""
    if schedules.empty:
        raise ValueError("Schedules frame is empty.")
    pbp_team_parts: list[pd.DataFrame] = []
    qb_parts: list[pd.DataFrame] = []
    for season in sorted(pbp_by_season):
        pbp = pbp_by_season[season]
        if pbp.empty:
            raise ValueError(f"PBP frame for {season} is empty; refusing silent substitution.")
        pbp_team_parts.append(team_game_metrics(pbp))
        qb_parts.append(qb_game_metrics(pbp))

    pbp_team = pd.concat(pbp_team_parts, ignore_index=True) if pbp_team_parts else pd.DataFrame()
    qb_games = pd.concat(qb_parts, ignore_index=True) if qb_parts else pd.DataFrame()
    schedule_games = schedule_team_game_metrics(schedules)
    team_games = combine_team_game_metrics(pbp_team, schedule_games)
    team_pregame = add_pregame_rolling_features(team_games)
    qb_pregame = build_pregame_qb_features(schedules, qb_games)
    team_pregame = _attach_qb_features(team_pregame, qb_pregame)
    return build_matchup_frame(team_pregame, schedules)


def build_historical_backtest_frame(
    schedules: pd.DataFrame,
    pbp_by_season: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    """Build fair-line features first, then join nflverse closing spread afterward."""
    fair = build_feature_frame_from_frames(schedules, pbp_by_season)
    if "spread_line" not in schedules.columns:
        raise ValueError("Historical schedule data is missing nflverse spread_line.")
    market = schedules[["game_id", "spread_line"]].drop_duplicates("game_id")
    return fair.merge(market, on="game_id", how="left", validate="one_to_one")


def load_and_build(
    seasons: list[int] | tuple[int, ...] | None = None,
    cache_dir: str | Path = "data/cache",
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
) -> pd.DataFrame:
    requested = tuple(seasons or (config.training_seasons + config.validation_seasons + config.holdout_seasons))
    schedules, _ = load_schedules(cache_dir=cache_dir, config=config, refresh=refresh)
    schedules = schedules[schedules["season"].isin(requested)].copy()
    missing_seasons = sorted(set(requested) - set(int(x) for x in schedules["season"].dropna().unique()))
    if missing_seasons:
        raise ValueError(f"Schedule data missing requested seasons: {missing_seasons}")
    pbp_by_season: dict[int, pd.DataFrame] = {}
    for season in requested:
        pbp, _ = load_pbp(season, cache_dir=cache_dir, config=config, refresh=refresh)
        pbp_by_season[int(season)] = pbp
    return build_historical_backtest_frame(schedules, pbp_by_season)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BEARD market-blind pregame NFL feature frame from nflverse")
    parser.add_argument("--seasons", nargs="+", type=int, help="Seasons to load; defaults to configured 2015-2025 split")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output", default="artifacts/beard_matchups.parquet")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    frame = load_and_build(args.seasons, args.cache_dir, DEFAULT_CONFIG, args.refresh)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    else:
        frame.to_parquet(output, index=False)
    print(f"Wrote {len(frame):,} matchup rows to {output}")
    print("Market integrity: nflverse spread_line was joined only after market-blind matchup features were built.")


if __name__ == "__main__":
    main()
