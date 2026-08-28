from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .integrity import assert_market_blind_features
from .qualification import grade_side, nflverse_spread_to_market_home_spread


RIDGE_ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0)
DEV_VALIDATION_SEASONS = (2020, 2021, 2022, 2023, 2024)
RETROSPECTIVE_AUDIT_SEASON = 2025


@dataclass(frozen=True)
class V02Spec:
    rating_k: float = 0.22
    offseason_regression: float = 0.55
    home_advantage_points: float = 1.5
    alpha_candidates: tuple[float, ...] = RIDGE_ALPHAS
    development_validation_seasons: tuple[int, ...] = DEV_VALIDATION_SEASONS
    retrospective_audit_season: int = RETROSPECTIVE_AUDIT_SEASON


CORE_SOURCE_COLUMNS = (
    "home_field", "rest_diff",
    "home_pre_games_played", "away_pre_games_played",
    "home_pre_margin_r4", "away_pre_margin_r4", "home_pre_margin_r8", "away_pre_margin_r8",
    "home_pre_epa_per_play_r4", "away_pre_epa_per_play_r4", "home_pre_epa_per_play_r8", "away_pre_epa_per_play_r8",
    "home_pre_epa_per_play_allowed_r8", "away_pre_epa_per_play_allowed_r8",
    "home_pre_success_rate_r8", "away_pre_success_rate_r8",
    "home_pre_success_rate_allowed_r8", "away_pre_success_rate_allowed_r8",
    "home_pre_pass_epa_per_play_r4", "away_pre_pass_epa_per_play_r4",
    "home_pre_pass_epa_per_play_r8", "away_pre_pass_epa_per_play_r8",
    "home_pre_pass_epa_per_play_allowed_r8", "away_pre_pass_epa_per_play_allowed_r8",
    "home_pre_rush_epa_per_play_r8", "away_pre_rush_epa_per_play_r8",
    "home_pre_rush_epa_per_play_allowed_r8", "away_pre_rush_epa_per_play_allowed_r8",
    "home_pre_explosive_pass_rate_r8", "away_pre_explosive_pass_rate_r8",
    "home_pre_explosive_pass_rate_allowed_r8", "away_pre_explosive_pass_rate_allowed_r8",
    "home_pre_early_down_epa_r8", "away_pre_early_down_epa_r8",
    "home_pre_early_down_epa_allowed_r8", "away_pre_early_down_epa_allowed_r8",
    "home_pre_sack_rate_r8", "away_pre_sack_rate_r8",
    "home_pre_sack_rate_generated_r8", "away_pre_sack_rate_generated_r8",
    "home_pre_interception_rate_r8", "away_pre_interception_rate_r8",
    "home_pre_interception_rate_generated_r8", "away_pre_interception_rate_generated_r8",
    "home_pre_qb_epa_per_dropback_r4", "away_pre_qb_epa_per_dropback_r4",
    "home_pre_qb_epa_per_dropback_r8", "away_pre_qb_epa_per_dropback_r8",
    "home_pre_qb_success_rate_r8", "away_pre_qb_success_rate_r8",
    "home_pre_qb_cpoe_r8", "away_pre_qb_cpoe_r8",
    "home_pre_qb_sack_rate_r8", "away_pre_qb_sack_rate_r8",
    "home_pre_qb_starts", "away_pre_qb_starts",
    "home_starter_change", "away_starter_change",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("v0.2 source frame missing required columns: " + ", ".join(missing))


def add_dynamic_margin_rating(frame: pd.DataFrame, spec: V02Spec = V02Spec()) -> pd.DataFrame:
    """Market-blind opponent-adjusted points rating, frozen before each game."""
    required = {"season", "gameday", "game_id", "home_team", "away_team", "home_margin", "home_field"}
    _require_columns(frame, required)
    out = frame.sort_values(["gameday", "game_id"]).copy()
    ratings: dict[str, float] = {}
    last_season: int | None = None
    pre_home: list[float] = []
    pre_away: list[float] = []
    pre_diff: list[float] = []
    pre_prediction: list[float] = []

    for row in out.itertuples(index=False):
        season = int(row.season)
        if last_season is None:
            last_season = season
        elif season != last_season:
            ratings = {team: value * spec.offseason_regression for team, value in ratings.items()}
            last_season = season

        h = float(ratings.get(row.home_team, 0.0))
        a = float(ratings.get(row.away_team, 0.0))
        hfa = spec.home_advantage_points * float(row.home_field)
        prediction = h - a + hfa
        pre_home.append(h)
        pre_away.append(a)
        pre_diff.append(h - a)
        pre_prediction.append(prediction)

        if pd.notna(row.home_margin):
            residual = float(np.clip(float(row.home_margin) - prediction, -28.0, 28.0))
            step = spec.rating_k * residual / 2.0
            ratings[row.home_team] = h + step
            ratings[row.away_team] = a - step

    out["v02_home_dynamic_rating"] = pre_home
    out["v02_away_dynamic_rating"] = pre_away
    out["v02_dynamic_rating_diff"] = pre_diff
    out["v02_dynamic_margin_baseline"] = pre_prediction
    return out.sort_index()


def add_v02_features(frame: pd.DataFrame, spec: V02Spec = V02Spec()) -> pd.DataFrame:
    _require_columns(frame, CORE_SOURCE_COLUMNS)
    x = add_dynamic_margin_rating(frame, spec)
    x["v02_matchup_epa"] = x["home_pre_epa_per_play_r8"] + x["away_pre_epa_per_play_allowed_r8"] - x["away_pre_epa_per_play_r8"] - x["home_pre_epa_per_play_allowed_r8"]
    x["v02_matchup_pass_epa"] = x["home_pre_pass_epa_per_play_r8"] + x["away_pre_pass_epa_per_play_allowed_r8"] - x["away_pre_pass_epa_per_play_r8"] - x["home_pre_pass_epa_per_play_allowed_r8"]
    x["v02_matchup_rush_epa"] = x["home_pre_rush_epa_per_play_r8"] + x["away_pre_rush_epa_per_play_allowed_r8"] - x["away_pre_rush_epa_per_play_r8"] - x["home_pre_rush_epa_per_play_allowed_r8"]
    x["v02_matchup_success"] = x["home_pre_success_rate_r8"] + x["away_pre_success_rate_allowed_r8"] - x["away_pre_success_rate_r8"] - x["home_pre_success_rate_allowed_r8"]
    x["v02_matchup_early_down"] = x["home_pre_early_down_epa_r8"] + x["away_pre_early_down_epa_allowed_r8"] - x["away_pre_early_down_epa_r8"] - x["home_pre_early_down_epa_allowed_r8"]
    x["v02_matchup_explosive_pass"] = x["home_pre_explosive_pass_rate_r8"] + x["away_pre_explosive_pass_rate_allowed_r8"] - x["away_pre_explosive_pass_rate_r8"] - x["home_pre_explosive_pass_rate_allowed_r8"]
    x["v02_sack_matchup"] = x["home_pre_sack_rate_generated_r8"] + x["away_pre_sack_rate_r8"] - x["away_pre_sack_rate_generated_r8"] - x["home_pre_sack_rate_r8"]
    x["v02_interception_matchup"] = x["home_pre_interception_rate_generated_r8"] + x["away_pre_interception_rate_r8"] - x["away_pre_interception_rate_generated_r8"] - x["home_pre_interception_rate_r8"]
    x["v02_recent_margin_diff"] = x["home_pre_margin_r8"] - x["away_pre_margin_r8"]
    x["v02_margin_acceleration"] = x["home_pre_margin_r4"] - x["home_pre_margin_r8"] - x["away_pre_margin_r4"] + x["away_pre_margin_r8"]
    x["v02_epa_acceleration"] = x["home_pre_epa_per_play_r4"] - x["home_pre_epa_per_play_r8"] - x["away_pre_epa_per_play_r4"] + x["away_pre_epa_per_play_r8"]
    x["v02_pass_epa_acceleration"] = x["home_pre_pass_epa_per_play_r4"] - x["home_pre_pass_epa_per_play_r8"] - x["away_pre_pass_epa_per_play_r4"] + x["away_pre_pass_epa_per_play_r8"]
    x["v02_qb_epa_diff"] = x["home_pre_qb_epa_per_dropback_r8"] - x["away_pre_qb_epa_per_dropback_r8"]
    x["v02_qb_epa_acceleration"] = x["home_pre_qb_epa_per_dropback_r4"] - x["home_pre_qb_epa_per_dropback_r8"] - x["away_pre_qb_epa_per_dropback_r4"] + x["away_pre_qb_epa_per_dropback_r8"]
    x["v02_qb_success_diff"] = x["home_pre_qb_success_rate_r8"] - x["away_pre_qb_success_rate_r8"]
    x["v02_qb_cpoe_diff"] = x["home_pre_qb_cpoe_r8"] - x["away_pre_qb_cpoe_r8"]
    x["v02_qb_sack_diff"] = x["home_pre_qb_sack_rate_r8"] - x["away_pre_qb_sack_rate_r8"]
    x["v02_qb_experience_log_diff"] = np.log1p(x["home_pre_qb_starts"].clip(lower=0)) - np.log1p(x["away_pre_qb_starts"].clip(lower=0))
    x["v02_starter_change_diff"] = x["home_starter_change"] - x["away_starter_change"]
    x["v02_min_games_played"] = x[["home_pre_games_played", "away_pre_games_played"]].min(axis=1)
    return x


V02_FEATURES = (
    "v02_dynamic_margin_baseline", "home_field", "rest_diff", "v02_matchup_epa", "v02_matchup_pass_epa",
    "v02_matchup_rush_epa", "v02_matchup_success", "v02_matchup_early_down", "v02_matchup_explosive_pass",
    "v02_sack_matchup", "v02_interception_matchup", "v02_recent_margin_diff", "v02_margin_acceleration",
    "v02_epa_acceleration", "v02_pass_epa_acceleration", "v02_qb_epa_diff", "v02_qb_epa_acceleration",
    "v02_qb_success_diff", "v02_qb_cpoe_diff", "v02_qb_sack_diff", "v02_qb_experience_log_diff",
    "v02_starter_change_diff", "v02_min_games_played",
)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))])


def select_alpha_walk_forward(frame: pd.DataFrame, spec: V02Spec = V02Spec()) -> dict:
    """Choose regularization on 2020-2024 walk-forward MAE only; never inspect 2025 here."""
    x = add_v02_features(frame, spec)
    assert_market_blind_features(V02_FEATURES)
    rows = []
    for alpha in spec.alpha_candidates:
        fold_details = []
        for val_season in spec.development_validation_seasons:
            train = x[(x["season"] < val_season) & x["home_margin"].notna()].copy()
            val = x[(x["season"] == val_season) & x["home_margin"].notna()].copy()
            if train.empty or val.empty:
                raise ValueError(f"Missing train/validation rows for {val_season}")
            model = ridge_pipeline(alpha)
            model.fit(train[list(V02_FEATURES)], train["home_margin"])
            pred = model.predict(val[list(V02_FEATURES)])
            fold_details.append({"season": int(val_season), "mae": float(mean_absolute_error(val["home_margin"], pred)), "games": int(len(val))})
        weighted = float(np.average([d["mae"] for d in fold_details], weights=[d["games"] for d in fold_details]))
        rows.append({"alpha": float(alpha), "weighted_mae": weighted, "folds": fold_details})
    best = min(rows, key=lambda r: (r["weighted_mae"], r["alpha"]))
    return {"selected_alpha": best["alpha"], "candidates": rows, "selected_using_seasons": list(spec.development_validation_seasons)}


def _ats_rows(test: pd.DataFrame, fair_margin: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame(index=test.index)
    rows["home_margin"] = test["home_margin"].astype(float)
    rows["market_home_spread"] = test["spread_line"].astype(float).map(nflverse_spread_to_market_home_spread)
    rows["fair_margin"] = fair_margin
    rows["edge_signed_home"] = rows["fair_margin"] + rows["market_home_spread"]
    rows["edge_abs"] = rows["edge_signed_home"].abs()
    rows["side"] = np.where(rows["edge_signed_home"] > 0, "HOME", np.where(rows["edge_signed_home"] < 0, "AWAY", "NONE"))
    rows["ats_result"] = [grade_side(m, s, side) for m, s, side in zip(rows["home_margin"], rows["market_home_spread"], rows["side"])]
    return rows


def _ats_summary(rows: pd.DataFrame, threshold: float) -> dict:
    sub = rows[rows["edge_abs"] >= threshold]
    wins = int((sub["ats_result"] == "WIN").sum()); losses = int((sub["ats_result"] == "LOSS").sum()); pushes = int((sub["ats_result"] == "PUSH").sum())
    n = wins + losses; net = wins * (100 / 110) - losses
    return {"threshold": float(threshold), "bets": n, "wins": wins, "losses": losses, "pushes": pushes, "win_pct": None if n == 0 else wins/n, "roi": None if n == 0 else net/n, "coverage_pct": 0.0 if len(rows) == 0 else 100*n/len(rows)}


def evaluate_walk_forward_v02(frame: pd.DataFrame, spec: V02Spec = V02Spec(), thresholds=(1,2,3,4,5)) -> dict:
    """Evaluate fixed v0.2 architecture. 2025 is retrospective audit, not untouched holdout."""
    x = add_v02_features(frame, spec)
    alpha_result = select_alpha_walk_forward(frame, spec)
    alpha = alpha_result["selected_alpha"]
    folds = []
    for season in (*spec.development_validation_seasons, spec.retrospective_audit_season):
        train = x[(x["season"] < season) & x["home_margin"].notna()].copy()
        test = x[(x["season"] == season) & x["home_margin"].notna()].copy()
        model = ridge_pipeline(alpha)
        model.fit(train[list(V02_FEATURES)], train["home_margin"])
        pred = model.predict(test[list(V02_FEATURES)])
        y = test["home_margin"].to_numpy(float); market = test["spread_line"].to_numpy(float)
        ats = _ats_rows(test, pred)
        folds.append({"season": int(season), "role": "development_walk_forward" if season in spec.development_validation_seasons else "retrospective_audit_not_pristine", "games": int(len(test)), "beard_mae": float(mean_absolute_error(y, pred)), "market_mae": float(mean_absolute_error(y, market)), "beard_minus_market_mae": float(mean_absolute_error(y, pred)-mean_absolute_error(y, market)), "beard_rmse": float(np.sqrt(mean_squared_error(y, pred))), "market_rmse": float(np.sqrt(mean_squared_error(y, market))), "ats": [_ats_summary(ats, float(t)) for t in thresholds]})
    aggregate_dev = []
    for t in thresholds:
        selected = [next(a for a in f["ats"] if a["threshold"] == float(t)) for f in folds if f["role"] == "development_walk_forward"]
        wins=sum(a["wins"] for a in selected); losses=sum(a["losses"] for a in selected); pushes=sum(a["pushes"] for a in selected); bets=wins+losses; net=wins*(100/110)-losses
        aggregate_dev.append({"threshold":float(t),"bets":bets,"wins":wins,"losses":losses,"pushes":pushes,"win_pct":None if not bets else wins/bets,"roi":None if not bets else net/bets})
    return {"label":"BEARD_V0_2_DEVELOPMENT_DIAGNOSTIC", "feature_count":len(V02_FEATURES), "features":list(V02_FEATURES), "alpha_selection":alpha_result, "folds":folds, "aggregate_development_2020_2024":aggregate_dev, "scientific_status":"2025 has already been observed during v0.1 development, so it is a retrospective audit, not a pristine holdout. True prospective confirmation must be 2026 forward or another predeclared unseen sample."}
