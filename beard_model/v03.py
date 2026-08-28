from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .qualification import grade_side, nflverse_spread_to_market_home_spread
from .v02 import add_v02_features


# v0.3 is intentionally POST-MARKET. It models the error in an already-observed
# market fair margin. It must never be represented as the market-blind BEARD fair line.
RESIDUAL_RIDGE_ALPHA = 300.0
COVER_LOGISTIC_C = 0.05
QUALIFIED_PROBABILITY = 0.55
QUALIFIED_RESIDUAL_POINTS = 1.5
STRONG_PROBABILITY = 0.57
STRONG_RESIDUAL_POINTS = 2.0


V03_FEATURES = (
    "market_fair_margin",
    "market_abs_spread",
    "v03_dynamic_edge_to_market",
    "v02_matchup_epa",
    "v02_matchup_pass_epa",
    "v02_matchup_rush_epa",
    "v02_matchup_success",
    "v02_matchup_early_down",
    "v02_matchup_explosive_pass",
    "v02_sack_matchup",
    "v02_interception_matchup",
    "v02_qb_epa_diff",
    "v02_qb_success_diff",
    "v02_qb_cpoe_diff",
    "v02_qb_sack_diff",
    "v02_qb_experience_log_diff",
    "v02_starter_change_diff",
    "rest_diff",
    "home_field",
    "week",
    "v02_min_games_played",
)


@dataclass(frozen=True)
class V03Policy:
    residual_alpha: float = RESIDUAL_RIDGE_ALPHA
    logistic_c: float = COVER_LOGISTIC_C
    qualified_probability: float = QUALIFIED_PROBABILITY
    qualified_residual_points: float = QUALIFIED_RESIDUAL_POINTS
    strong_probability: float = STRONG_PROBABILITY
    strong_residual_points: float = STRONG_RESIDUAL_POINTS
    first_test_season: int = 2020
    last_test_season: int = 2025


def add_v03_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "spread_line" not in frame.columns:
        raise ValueError("v0.3 requires nflverse spread_line because it is a post-market residual model.")
    x = add_v02_features(frame)
    x["market_fair_margin"] = x["spread_line"].astype(float)
    x["market_abs_spread"] = x["market_fair_margin"].abs()
    x["v03_dynamic_edge_to_market"] = x["v02_dynamic_margin_baseline"] - x["market_fair_margin"]
    return x


def residual_target(frame: pd.DataFrame) -> pd.Series:
    """Positive means the home team outperformed the market fair margin."""
    if "home_margin" not in frame.columns or "spread_line" not in frame.columns:
        raise ValueError("Residual target needs home_margin and nflverse spread_line.")
    return frame["home_margin"].astype(float) - frame["spread_line"].astype(float)


def home_cover_binary(frame: pd.DataFrame) -> pd.Series:
    residual = residual_target(frame)
    return pd.Series(np.where(residual > 0, 1.0, np.where(residual < 0, 0.0, np.nan)), index=frame.index)


def residual_pipeline(alpha: float = RESIDUAL_RIDGE_ALPHA) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def cover_pipeline(c: float = COVER_LOGISTIC_C) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=float(c), solver="lbfgs", max_iter=3000)),
    ])


def qualify_predictions(
    cover_probability_home: np.ndarray,
    predicted_residual: np.ndarray,
    policy: V03Policy = V03Policy(),
) -> pd.DataFrame:
    p = np.asarray(cover_probability_home, dtype=float)
    r = np.asarray(predicted_residual, dtype=float)
    if len(p) != len(r):
        raise ValueError("Probability and residual arrays must have equal length.")

    home_agree = (p > 0.5) & (r > 0)
    away_agree = (p < 0.5) & (r < 0)
    side = np.where(home_agree, "HOME", np.where(away_agree, "AWAY", "NONE"))
    chosen_prob = np.where(side == "HOME", p, np.where(side == "AWAY", 1.0 - p, 0.5))
    abs_resid = np.abs(r)

    qualified = (
        (side != "NONE")
        & (chosen_prob >= policy.qualified_probability)
        & (abs_resid >= policy.qualified_residual_points)
    )
    strong = (
        (side != "NONE")
        & (chosen_prob >= policy.strong_probability)
        & (abs_resid >= policy.strong_residual_points)
    )
    status = np.where(strong, "STRONG", np.where(qualified, "QUALIFIED", "NO_PLAY"))
    return pd.DataFrame({
        "side": side,
        "chosen_probability": chosen_prob,
        "predicted_residual": r,
        "abs_predicted_residual": abs_resid,
        "status": status,
    })


def _grade(test: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    out["season"] = test["season"].to_numpy(int)
    out["week"] = test["week"].to_numpy(int)
    out["actual_home_margin"] = test["home_margin"].to_numpy(float)
    out["market_fair_margin"] = test["spread_line"].to_numpy(float)
    out["market_home_spread"] = np.array([
        nflverse_spread_to_market_home_spread(v) for v in out["market_fair_margin"]
    ])
    out["ats_result"] = [
        grade_side(m, s, side)
        for m, s, side in zip(out["actual_home_margin"], out["market_home_spread"], out["side"])
    ]
    return out


def _tier_summary(graded: pd.DataFrame, tier: str) -> dict:
    if tier == "QUALIFIED_OR_STRONG":
        sub = graded[graded["status"].isin(["QUALIFIED", "STRONG"])]
    else:
        sub = graded[graded["status"].eq(tier)]
    wins = int((sub["ats_result"] == "WIN").sum())
    losses = int((sub["ats_result"] == "LOSS").sum())
    pushes = int((sub["ats_result"] == "PUSH").sum())
    bets = wins + losses
    net = wins * (100.0 / 110.0) - losses
    return {
        "tier": tier,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": None if bets == 0 else wins / bets,
        "roi": None if bets == 0 else net / bets,
        "coverage_pct": 0.0 if len(graded) == 0 else 100.0 * bets / len(graded),
    }


def evaluate_v03(frame: pd.DataFrame, policy: V03Policy = V03Policy()) -> dict:
    """Expanding walk-forward evaluation. Hyperparameters/policy are fixed before the run."""
    x = add_v03_features(frame)
    folds = []
    all_graded = []

    for season in range(policy.first_test_season, policy.last_test_season + 1):
        train = x[(x["season"] < season) & x["home_margin"].notna()].copy()
        test = x[(x["season"] == season) & x["home_margin"].notna()].copy()
        if train.empty or test.empty:
            raise ValueError(f"Missing train/test rows for {season}.")

        y_resid = residual_target(train)
        y_cover = home_cover_binary(train)
        class_mask = y_cover.notna()

        resid_model = residual_pipeline(policy.residual_alpha)
        resid_model.fit(train[list(V03_FEATURES)], y_resid)
        pred_resid = resid_model.predict(test[list(V03_FEATURES)])

        cover_model = cover_pipeline(policy.logistic_c)
        cover_model.fit(train.loc[class_mask, list(V03_FEATURES)], y_cover.loc[class_mask].astype(int))
        p_home = cover_model.predict_proba(test[list(V03_FEATURES)])[:, 1]

        q = qualify_predictions(p_home, pred_resid, policy)
        graded = _grade(test.reset_index(drop=True), q.reset_index(drop=True))
        all_graded.append(graded)

        actual_resid = residual_target(test).to_numpy(float)
        anchored_margin = test["spread_line"].to_numpy(float) + pred_resid
        market_margin = test["spread_line"].to_numpy(float)
        y_margin = test["home_margin"].to_numpy(float)
        cover_truth = home_cover_binary(test)
        nonpush = cover_truth.notna().to_numpy()
        probs = np.clip(p_home[nonpush], 1e-6, 1 - 1e-6)
        truth = cover_truth.to_numpy()[nonpush].astype(int)

        folds.append({
            "season": int(season),
            "games": int(len(test)),
            "market_margin_mae": float(mean_absolute_error(y_margin, market_margin)),
            "anchored_margin_mae": float(mean_absolute_error(y_margin, anchored_margin)),
            "anchored_mae_minus_market": float(mean_absolute_error(y_margin, anchored_margin) - mean_absolute_error(y_margin, market_margin)),
            "market_margin_rmse": float(np.sqrt(mean_squared_error(y_margin, market_margin))),
            "anchored_margin_rmse": float(np.sqrt(mean_squared_error(y_margin, anchored_margin))),
            "residual_mae": float(mean_absolute_error(actual_resid, pred_resid)),
            "cover_brier": float(brier_score_loss(truth, probs)),
            "cover_log_loss": float(log_loss(truth, probs, labels=[0, 1])),
            "qualified_or_strong": _tier_summary(graded, "QUALIFIED_OR_STRONG"),
            "strong": _tier_summary(graded, "STRONG"),
        })

    combined = pd.concat(all_graded, ignore_index=True)
    return {
        "label": "BEARD_V0_3_MARKET_RESIDUAL_DEVELOPMENT",
        "architecture": "post-market residual Ridge + home-cover logistic; both must agree on side",
        "features": list(V03_FEATURES),
        "fixed_policy": {
            "residual_alpha": policy.residual_alpha,
            "logistic_c": policy.logistic_c,
            "qualified_probability": policy.qualified_probability,
            "qualified_residual_points": policy.qualified_residual_points,
            "strong_probability": policy.strong_probability,
            "strong_residual_points": policy.strong_residual_points,
        },
        "folds": folds,
        "aggregate_2020_2025": {
            "qualified_or_strong": _tier_summary(combined, "QUALIFIED_OR_STRONG"),
            "strong": _tier_summary(combined, "STRONG"),
        },
        "scientific_status": (
            "This is historical walk-forward development on data already inspected in prior BEARD iterations. "
            "It can reject architectures but cannot establish a pristine betting edge. A frozen model must be "
            "paper-traded prospectively in 2026 before any performance claim."
        ),
    }
