from __future__ import annotations

import numpy as np
import pandas as pd

from .integrity import assert_market_blind_features


def _prefixed(row: pd.Series, prefix: str, columns: list[str]) -> dict[str, float]:
    return {f"{prefix}{c}": row.get(c, np.nan) for c in columns}


def build_matchup_frame(team_pregame: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Build one market-blind row per game from strictly pregame team snapshots."""
    required_team = {"game_id", "team", "opponent", "season", "week"}
    missing = sorted(required_team - set(team_pregame.columns))
    if missing:
        raise ValueError(f"team_pregame missing columns: {', '.join(missing)}")
    required_sched = {
        "game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score",
        "home_rest", "away_rest", "location",
    }
    missing = sorted(required_sched - set(schedules.columns))
    if missing:
        raise ValueError(f"schedules missing columns: {', '.join(missing)}")

    pre_cols = [c for c in team_pregame.columns if c.startswith("pre_") or c in {"starter_change"}]
    assert_market_blind_features(pre_cols)
    lookup = team_pregame.set_index(["game_id", "team"], drop=False)
    rows: list[dict] = []
    for s in schedules.itertuples(index=False):
        key_h, key_a = (s.game_id, s.home_team), (s.game_id, s.away_team)
        if key_h not in lookup.index or key_a not in lookup.index:
            continue
        h = lookup.loc[key_h]
        a = lookup.loc[key_a]
        if isinstance(h, pd.DataFrame):
            h = h.iloc[0]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        neutral = str(s.location).lower() == "neutral"
        row = {
            "game_id": s.game_id, "season": int(s.season), "week": int(s.week), "gameday": pd.Timestamp(s.gameday),
            "home_team": s.home_team, "away_team": s.away_team,
            "home_field": 0.0 if neutral else 1.0,
            "rest_diff": (float(s.home_rest) - float(s.away_rest)) if pd.notna(s.home_rest) and pd.notna(s.away_rest) else np.nan,
            "home_margin": (float(s.home_score) - float(s.away_score)) if pd.notna(s.home_score) and pd.notna(s.away_score) else np.nan,
            "total_points": (float(s.home_score) + float(s.away_score)) if pd.notna(s.home_score) and pd.notna(s.away_score) else np.nan,
        }
        row.update(_prefixed(h, "home_", pre_cols))
        row.update(_prefixed(a, "away_", pre_cols))
        for c in pre_cols:
            hv, av = row.get(f"home_{c}"), row.get(f"away_{c}")
            if pd.api.types.is_number(hv) or isinstance(hv, (int, float, np.number)):
                try:
                    row[f"diff_{c}"] = float(hv) - float(av)
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    pairs = {
        "home_pass_matchup_r4": ("home_pre_pass_epa_per_play_r4", "away_pre_pass_epa_per_play_allowed_r4"),
        "away_pass_matchup_r4": ("away_pre_pass_epa_per_play_r4", "home_pre_pass_epa_per_play_allowed_r4"),
        "home_rush_matchup_r4": ("home_pre_rush_epa_per_play_r4", "away_pre_rush_epa_per_play_allowed_r4"),
        "away_rush_matchup_r4": ("away_pre_rush_epa_per_play_r4", "home_pre_rush_epa_per_play_allowed_r4"),
        "home_explosive_pass_matchup_r4": ("home_pre_explosive_pass_rate_r4", "away_pre_explosive_pass_rate_allowed_r4"),
        "away_explosive_pass_matchup_r4": ("away_pre_explosive_pass_rate_r4", "home_pre_explosive_pass_rate_allowed_r4"),
        "home_protection_vs_rush_r4": ("away_pre_sack_rate_generated_r4", "home_pre_sack_rate_r4"),
        "away_protection_vs_rush_r4": ("home_pre_sack_rate_generated_r4", "away_pre_sack_rate_r4"),
    }
    for name, (left, right) in pairs.items():
        if left in out.columns and right in out.columns:
            out[name] = out[left] - out[right]

    model_feature_cols = [c for c in out.columns if c not in {"game_id", "season", "week", "gameday", "home_team", "away_team", "home_margin", "total_points"}]
    assert_market_blind_features(model_feature_cols)
    return out


def model_feature_columns(matchups: pd.DataFrame) -> list[str]:
    excluded = {"game_id", "season", "week", "gameday", "home_team", "away_team", "home_margin", "total_points"}
    cols = [c for c in matchups.columns if c not in excluded and pd.api.types.is_numeric_dtype(matchups[c])]
    assert_market_blind_features(cols)
    return cols
