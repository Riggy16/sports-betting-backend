from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .qualification import grade_side, nflverse_spread_to_market_home_spread
from .v02 import V02_FEATURES, add_v02_features, ridge_pipeline


FAIR_ALPHA = 300.0
META_START_SEASON = 2017
QUALIFIER_C = 0.10
ELIGIBLE_MIN_EDGE = 1.0
PROBABILITY_GATES = (0.525, 0.54, 0.55, 0.56, 0.58, 0.60)


Q_FEATURES = (
    "edge_abs",
    "market_abs_spread",
    "selected_side_home",
    "selected_side_favorite",
    "week",
    "min_games_played",
    "selected_qb_change",
    "selected_qb_starts_log",
    "opponent_qb_starts_log",
    "fair_abs_margin",
    "home_field",
    "abs_rest_diff",
    "prior_year_fair_resid_std",
    "edge_z_prior_error",
)


@dataclass(frozen=True)
class QV2Spec:
    fair_alpha: float = FAIR_ALPHA
    meta_start_season: int = META_START_SEASON
    qualifier_c: float = QUALIFIER_C
    eligible_min_edge: float = ELIGIBLE_MIN_EDGE
    probability_gates: tuple[float, ...] = PROBABILITY_GATES
    development_test_seasons: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
    retrospective_audit_season: int = 2025


def qualifier_pipeline(c: float = QUALIFIER_C) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=float(c), solver="lbfgs", max_iter=3000)),
    ])


def _prior_year_error_std(x: pd.DataFrame, season: int, alpha: float) -> float:
    """Estimate fair-line error using only the season immediately before `season`."""
    prior = int(season) - 1
    train = x[(x["season"] < prior) & x["home_margin"].notna()]
    val = x[(x["season"] == prior) & x["home_margin"].notna()]
    if train.empty or val.empty:
        return float("nan")
    model = ridge_pipeline(alpha)
    model.fit(train[list(V02_FEATURES)], train["home_margin"])
    pred = model.predict(val[list(V02_FEATURES)])
    resid = val["home_margin"].to_numpy(float) - pred
    if len(resid) < 2:
        return float("nan")
    return float(np.std(resid, ddof=1))


def build_walk_forward_meta(frame: pd.DataFrame, spec: QV2Spec = QV2Spec()) -> pd.DataFrame:
    """Generate OOS fair predictions first, then attach post-market qualification features."""
    required = {"season", "home_margin", "spread_line", "week", "home_field", "rest_diff"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Qualification meta frame missing: " + ", ".join(missing))
    x = add_v02_features(frame)
    max_season = int(x["season"].max())
    rows: list[pd.DataFrame] = []
    for season in range(spec.meta_start_season, max_season + 1):
        train = x[(x["season"] < season) & x["home_margin"].notna()].copy()
        test = x[(x["season"] == season) & x["home_margin"].notna()].copy()
        if train.empty or test.empty:
            continue
        fair_model = ridge_pipeline(spec.fair_alpha)
        fair_model.fit(train[list(V02_FEATURES)], train["home_margin"])
        fair_margin = fair_model.predict(test[list(V02_FEATURES)])
        market_fair_margin = test["spread_line"].to_numpy(float)
        edge_signed_home = fair_margin - market_fair_margin
        side_home = edge_signed_home > 0
        edge_abs = np.abs(edge_signed_home)
        market_home_spread = np.array([nflverse_spread_to_market_home_spread(v) for v in market_fair_margin])
        side = np.where(side_home, "HOME", np.where(edge_signed_home < 0, "AWAY", "NONE"))
        result = [
            grade_side(actual, spread, chosen)
            for actual, spread, chosen in zip(test["home_margin"].to_numpy(float), market_home_spread, side)
        ]
        selected_favorite = np.where(side_home, market_fair_margin > 0, market_fair_margin < 0)
        selected_qb_change = np.where(side_home, test["home_starter_change"], test["away_starter_change"])
        selected_qb_starts = np.where(side_home, test["home_pre_qb_starts"], test["away_pre_qb_starts"])
        opponent_qb_starts = np.where(side_home, test["away_pre_qb_starts"], test["home_pre_qb_starts"])
        prior_std = _prior_year_error_std(x, season, spec.fair_alpha)

        meta = pd.DataFrame({
            "game_id": test["game_id"].to_numpy(),
            "season": int(season),
            "week": test["week"].to_numpy(float),
            "actual_home_margin": test["home_margin"].to_numpy(float),
            "market_fair_margin": market_fair_margin,
            "fair_margin": fair_margin,
            "edge_signed_home": edge_signed_home,
            "edge_abs": edge_abs,
            "market_abs_spread": np.abs(market_fair_margin),
            "selected_side_home": side_home.astype(float),
            "selected_side_favorite": selected_favorite.astype(float),
            "min_games_played": test["v02_min_games_played"].to_numpy(float),
            "selected_qb_change": np.asarray(selected_qb_change, dtype=float),
            "selected_qb_starts_log": np.log1p(np.maximum(np.asarray(selected_qb_starts, dtype=float), 0.0)),
            "opponent_qb_starts_log": np.log1p(np.maximum(np.asarray(opponent_qb_starts, dtype=float), 0.0)),
            "fair_abs_margin": np.abs(fair_margin),
            "home_field": test["home_field"].to_numpy(float),
            "abs_rest_diff": np.abs(test["rest_diff"].to_numpy(float)),
            "prior_year_fair_resid_std": prior_std,
            "ats_result": result,
        })
        meta["edge_z_prior_error"] = meta["edge_abs"] / meta["prior_year_fair_resid_std"].replace(0, np.nan)
        meta["ats_win"] = np.where(meta["ats_result"].eq("WIN"), 1.0, np.where(meta["ats_result"].eq("LOSS"), 0.0, np.nan))
        rows.append(meta)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _fixed_gate_summary(test: pd.DataFrame, probability: np.ndarray, gate: float) -> dict:
    chosen = test[np.asarray(probability) >= float(gate)]
    wins = int((chosen["ats_result"] == "WIN").sum())
    losses = int((chosen["ats_result"] == "LOSS").sum())
    pushes = int((chosen["ats_result"] == "PUSH").sum())
    bets = wins + losses
    net = wins * (100.0 / 110.0) - losses
    return {
        "probability_gate": float(gate),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": None if bets == 0 else wins / bets,
        "roi": None if bets == 0 else net / bets,
        "coverage_pct": 0.0 if len(test) == 0 else 100.0 * bets / len(test),
    }


def _raw_edge_summary(test: pd.DataFrame, threshold: float = 4.0) -> dict:
    chosen = test[test["edge_abs"] >= float(threshold)]
    wins = int((chosen["ats_result"] == "WIN").sum())
    losses = int((chosen["ats_result"] == "LOSS").sum())
    pushes = int((chosen["ats_result"] == "PUSH").sum())
    bets = wins + losses
    net = wins * (100.0 / 110.0) - losses
    return {
        "edge_threshold": float(threshold), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
        "win_pct": None if bets == 0 else wins / bets,
        "roi": None if bets == 0 else net / bets,
    }


def evaluate_qv2(frame: pd.DataFrame, spec: QV2Spec = QV2Spec()) -> dict:
    """Walk-forward post-market qualifier evaluation with fixed probability gates."""
    meta = build_walk_forward_meta(frame, spec)
    if meta.empty:
        raise ValueError("No walk-forward qualification meta rows were generated.")
    eligible = meta[(meta["edge_abs"] >= spec.eligible_min_edge) & meta["ats_win"].notna()].copy()
    folds = []
    for season in (*spec.development_test_seasons, spec.retrospective_audit_season):
        train = eligible[eligible["season"] < season].copy()
        test = eligible[eligible["season"] == season].copy()
        if train.empty or test.empty:
            raise ValueError(f"Missing qualifier train/test rows for {season}.")
        model = qualifier_pipeline(spec.qualifier_c)
        model.fit(train[list(Q_FEATURES)], train["ats_win"].astype(int))
        prob = model.predict_proba(test[list(Q_FEATURES)])[:, 1]
        clipped = np.clip(prob, 1e-6, 1 - 1e-6)
        folds.append({
            "season": int(season),
            "role": "development_walk_forward" if season in spec.development_test_seasons else "retrospective_audit_not_pristine",
            "eligible_games": int(len(test)),
            "brier": float(brier_score_loss(test["ats_win"].astype(int), prob)),
            "log_loss": float(log_loss(test["ats_win"].astype(int), clipped, labels=[0, 1])),
            "mean_predicted_probability": float(np.mean(prob)),
            "actual_win_rate": float(test["ats_win"].mean()),
            "probability_gates": [_fixed_gate_summary(test, prob, gate) for gate in spec.probability_gates],
            "raw_edge_4_baseline": _raw_edge_summary(test, 4.0),
        })

    aggregate = []
    for gate in spec.probability_gates:
        rows = [next(g for g in f["probability_gates"] if g["probability_gate"] == float(gate)) for f in folds if f["role"] == "development_walk_forward"]
        wins = sum(r["wins"] for r in rows)
        losses = sum(r["losses"] for r in rows)
        pushes = sum(r["pushes"] for r in rows)
        bets = wins + losses
        net = wins * (100 / 110) - losses
        aggregate.append({
            "probability_gate": float(gate), "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "win_pct": None if bets == 0 else wins / bets,
            "roi": None if bets == 0 else net / bets,
        })

    return {
        "label": "BEARD_QV2_POST_MARKET_QUALIFIER_DEVELOPMENT",
        "fair_model": {"version": "v0.2-ridge", "fixed_alpha": spec.fair_alpha},
        "qualifier": {
            "type": "regularized_logistic_regression",
            "C": spec.qualifier_c,
            "eligible_min_edge": spec.eligible_min_edge,
            "features": list(Q_FEATURES),
            "probability_gates": list(spec.probability_gates),
        },
        "meta_seasons": sorted(int(s) for s in meta["season"].unique()),
        "folds": folds,
        "aggregate_development_2020_2024": aggregate,
        "scientific_status": "2025 is retrospective, not pristine. No production claim is permitted until prospective 2026 paper-trading or another predeclared unseen sample.",
    }
