from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import PBP_REQUIRED


BASE_GAME_METRICS = [
    "epa_per_play",
    "success_rate",
    "pass_epa_per_play",
    "rush_epa_per_play",
    "explosive_pass_rate",
    "early_down_epa",
    "sack_rate",
    "interception_rate",
]


def _require(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def team_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate standard nflverse PBP into one offense/defense-neutral team-game row.

    The returned metrics describe the just-finished game. They must be passed through
    `add_pregame_rolling_features` before being used as model inputs.
    """
    _require(pbp, PBP_REQUIRED, "PBP")
    x = pbp.copy()
    x = x[x["posteam"].notna() & x["defteam"].notna()].copy()
    x = x[x["play_type"].isin(["pass", "run"])].copy()
    x["is_pass"] = x["play_type"].eq("pass")
    x["is_rush"] = x["play_type"].eq("run")
    x["is_early_down"] = x["down"].isin([1, 2])
    x["is_explosive_pass"] = x["is_pass"] & x["yards_gained"].ge(20)

    rows: list[dict] = []
    group_cols = ["game_id", "season", "week", "posteam", "defteam"]
    for (game_id, season, week, offense, defense), g in x.groupby(group_cols, dropna=False, sort=False):
        pass_plays = g[g["is_pass"]]
        rush_plays = g[g["is_rush"]]
        early = g[g["is_early_down"]]
        rows.append({
            "game_id": game_id,
            "season": int(season),
            "week": int(week),
            "team": offense,
            "opponent": defense,
            "epa_per_play": _mean_or_nan(g["epa"]),
            "success_rate": _mean_or_nan(g["success"]),
            "pass_epa_per_play": _mean_or_nan(pass_plays["epa"]),
            "rush_epa_per_play": _mean_or_nan(rush_plays["epa"]),
            "explosive_pass_rate": _mean_or_nan(pass_plays["is_explosive_pass"].astype(float)),
            "early_down_epa": _mean_or_nan(early["epa"]),
            "sack_rate": _mean_or_nan(pass_plays["sack"].fillna(0).astype(float)),
            "interception_rate": _mean_or_nan(pass_plays["interception"].fillna(0).astype(float)),
        })
    offense = pd.DataFrame(rows)
    if offense.empty:
        return offense

    defense = offense[["game_id", "team"] + BASE_GAME_METRICS].copy()
    defense = defense.rename(columns={
        "team": "opponent",
        "epa_per_play": "epa_per_play_allowed",
        "success_rate": "success_rate_allowed",
        "pass_epa_per_play": "pass_epa_per_play_allowed",
        "rush_epa_per_play": "rush_epa_per_play_allowed",
        "explosive_pass_rate": "explosive_pass_rate_allowed",
        "early_down_epa": "early_down_epa_allowed",
        "sack_rate": "sack_rate_generated",
        "interception_rate": "interception_rate_generated",
    })
    return offense.merge(defense, on=["game_id", "opponent"], how="left", validate="one_to_one")


def schedule_team_game_metrics(schedules: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score",
        "home_rest", "away_rest", "location", "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    }
    _require(schedules, required, "schedules")
    records: list[dict] = []
    for row in schedules.itertuples(index=False):
        neutral = str(row.location).lower() == "neutral"
        home_score = float(row.home_score) if pd.notna(row.home_score) else np.nan
        away_score = float(row.away_score) if pd.notna(row.away_score) else np.nan
        records.extend([
            {
                "game_id": row.game_id, "season": int(row.season), "week": int(row.week), "gameday": pd.Timestamp(row.gameday),
                "team": row.home_team, "opponent": row.away_team, "points_for": home_score,
                "points_against": away_score, "margin": home_score - away_score,
                "rest_days": float(row.home_rest) if pd.notna(row.home_rest) else np.nan, "is_home": 0.0 if neutral else 1.0,
                "is_neutral": float(neutral), "starter_qb_id": row.home_qb_id, "starter_qb_name": row.home_qb_name,
            },
            {
                "game_id": row.game_id, "season": int(row.season), "week": int(row.week), "gameday": pd.Timestamp(row.gameday),
                "team": row.away_team, "opponent": row.home_team, "points_for": away_score,
                "points_against": home_score, "margin": away_score - home_score,
                "rest_days": float(row.away_rest) if pd.notna(row.away_rest) else np.nan, "is_home": 0.0,
                "is_neutral": float(neutral), "starter_qb_id": row.away_qb_id, "starter_qb_name": row.away_qb_name,
            },
        ])
    return pd.DataFrame.from_records(records)


def combine_team_game_metrics(pbp_metrics: pd.DataFrame, schedule_metrics: pd.DataFrame) -> pd.DataFrame:
    if pbp_metrics.empty:
        return schedule_metrics.copy()
    return schedule_metrics.merge(
        pbp_metrics,
        on=["game_id", "season", "week", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )


def add_pregame_rolling_features(
    team_games: pd.DataFrame,
    windows: tuple[int, ...] = (4, 8),
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Create strictly pregame rolling features by shifting before rolling."""
    _require(team_games, {"team", "season", "week", "game_id", "gameday"}, "team_games")
    out = team_games.sort_values(["team", "gameday", "game_id"]).copy()
    if metrics is None:
        exclude = {"season", "week", "is_home", "is_neutral"}
        metrics = [c for c in out.select_dtypes(include=[np.number]).columns if c not in exclude]
    group = out.groupby("team", sort=False, group_keys=False)

    for metric in metrics:
        if metric not in out.columns:
            continue
        for window in windows:
            out[f"pre_{metric}_r{window}"] = group[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
        out[f"pre_{metric}_season"] = out.groupby(["team", "season"], sort=False)[metric].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

    out["pre_games_played"] = out.groupby(["team", "season"], sort=False).cumcount()
    out["pre_previous_starter_qb_id"] = group["starter_qb_id"].shift(1)
    out["starter_change"] = (
        out["pre_previous_starter_qb_id"].notna()
        & out["starter_qb_id"].notna()
        & out["pre_previous_starter_qb_id"].ne(out["starter_qb_id"])
    ).astype(float)
    return out.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
