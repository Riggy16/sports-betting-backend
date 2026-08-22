from __future__ import annotations

import numpy as np
import pandas as pd


QB_CORE = {"game_id", "season", "week", "posteam", "play_type", "epa", "success", "sack", "interception"}


def _passer_columns(pbp: pd.DataFrame) -> tuple[str, str]:
    id_col = next((c for c in ["passer_player_id", "passer_id"] if c in pbp.columns), None)
    name_col = next((c for c in ["passer_player_name", "passer"] if c in pbp.columns), None)
    if id_col is None or name_col is None:
        raise ValueError("PBP needs a reliable passer id/name column for QB features.")
    return id_col, name_col


def qb_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(QB_CORE - set(pbp.columns))
    if missing:
        raise ValueError(f"PBP missing required QB columns: {', '.join(missing)}")
    id_col, name_col = _passer_columns(pbp)
    x = pbp.copy()
    if "qb_dropback" in x.columns:
        x = x[x["qb_dropback"].fillna(0).eq(1)].copy()
    else:
        x = x[x["play_type"].eq("pass")].copy()
    x = x[x[id_col].notna()].copy()
    epa_col = "qb_epa" if "qb_epa" in x.columns else "epa"

    rows: list[dict] = []
    keys = ["game_id", "season", "week", "posteam", id_col, name_col]
    for key, g in x.groupby(keys, dropna=False, sort=False):
        game_id, season, week, team, qb_id, qb_name = key
        row = {
            "game_id": game_id, "season": int(season), "week": int(week), "team": team,
            "qb_id": qb_id, "qb_name": qb_name, "qb_epa_per_dropback": float(g[epa_col].mean()),
            "qb_success_rate": float(g["success"].mean()),
            "qb_sack_rate": float(g["sack"].fillna(0).mean()),
            "qb_interception_rate": float(g["interception"].fillna(0).mean()),
            "qb_dropbacks": int(len(g)),
        }
        if "cpoe" in g.columns:
            nonnull = g["cpoe"].dropna()
            row["qb_cpoe"] = float(nonnull.mean()) if len(nonnull) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _starter_schedule_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    starters = []
    for r in schedules.itertuples(index=False):
        starters.extend([
            {"game_id": r.game_id, "season": int(r.season), "week": int(r.week), "gameday": pd.Timestamp(r.gameday),
             "team": r.home_team, "qb_id": r.home_qb_id, "qb_name": r.home_qb_name},
            {"game_id": r.game_id, "season": int(r.season), "week": int(r.week), "gameday": pd.Timestamp(r.gameday),
             "team": r.away_team, "qb_id": r.away_qb_id, "qb_name": r.away_qb_name},
        ])
    return pd.DataFrame(starters).sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)


def build_pregame_qb_features(
    schedules: pd.DataFrame,
    qb_games: pd.DataFrame,
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    """Attach each scheduled starter's statistics using only QB games before kickoff."""
    required = {
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    }
    missing = sorted(required - set(schedules.columns))
    if missing:
        raise ValueError(f"schedules missing QB columns: {', '.join(missing)}")
    starters = _starter_schedule_rows(schedules)
    starters["previous_starter_qb_id"] = starters.groupby("team", sort=False)["qb_id"].shift(1)
    starters["starter_change"] = (
        starters["previous_starter_qb_id"].notna() & starters["qb_id"].notna()
        & starters["previous_starter_qb_id"].ne(starters["qb_id"])
    ).astype(float)
    starters["availability_confidence"] = np.nan
    if qb_games.empty:
        starters["pre_qb_starts"] = 0
        return starters

    dates = starters[["game_id", "team", "gameday"]].drop_duplicates(["game_id", "team"])
    hist = qb_games.merge(dates, on=["game_id", "team"], how="left", validate="many_to_one")
    hist = hist[hist["qb_id"].notna() & hist["gameday"].notna()].sort_values(["qb_id", "gameday", "game_id"])
    metrics = [c for c in ["qb_epa_per_dropback", "qb_success_rate", "qb_sack_rate", "qb_interception_rate", "qb_cpoe"] if c in hist.columns]
    for metric in metrics:
        for window in windows:
            hist[f"hist_{metric}_r{window}"] = hist.groupby("qb_id", sort=False)[metric].transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
    hist["hist_qb_starts"] = hist.groupby("qb_id", sort=False).cumcount() + 1
    snap_cols = [c for c in hist.columns if c.startswith("hist_")]
    snapshots = hist[["qb_id", "gameday"] + snap_cols].sort_values(["gameday", "qb_id"])

    left = starters.sort_values(["gameday", "qb_id"], na_position="last").copy()
    known = left[left["qb_id"].notna()].copy()
    unknown = left[left["qb_id"].isna()].copy()
    if len(known):
        known = pd.merge_asof(
            known.sort_values(["gameday", "qb_id"]),
            snapshots.sort_values(["gameday", "qb_id"]),
            on="gameday", by="qb_id", direction="backward", allow_exact_matches=False,
        )
    out = pd.concat([known, unknown], ignore_index=True, sort=False)
    rename = {c: "pre_" + c.removeprefix("hist_") for c in out.columns if c.startswith("hist_")}
    out = out.rename(columns=rename)
    if "pre_qb_starts" not in out.columns:
        out["pre_qb_starts"] = 0
    out["pre_qb_starts"] = out["pre_qb_starts"].fillna(0).astype(int)
    return out.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
