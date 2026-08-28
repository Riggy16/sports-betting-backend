from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .integrity import assert_market_blind_features
from .qualification import grade_side, nflverse_spread_to_market_home_spread
from .v02 import add_dynamic_margin_rating


DEVELOPMENT_SEASONS = (2020, 2021, 2022, 2023, 2024)
RETROSPECTIVE_AUDIT_SEASON = 2025
ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass(frozen=True)
class TeamGrades:
    offense: float
    defense: float
    quarterback: float
    pass_offense: float
    rush_offense: float
    pass_defense: float
    rush_defense: float
    explosiveness: float
    pressure: float
    ball_security: float
    recent_form: float
    availability: float


@dataclass(frozen=True)
class MatchupGrades:
    pass_matchup: float
    rush_matchup: float
    pressure_matchup: float
    explosive_matchup: float
    qb_matchup: float
    form_matchup: float
    availability_matchup: float
    rest_edge_days: float
    home_field: float


@dataclass(frozen=True)
class GradedFairLine:
    home_team: str
    away_team: str
    home_team_grade: float
    away_team_grade: float
    base_grade_edge: float
    matchup_grade_edge: float
    fair_home_margin: float
    projected_home_spread: float
    components: dict[str, float]


SOURCE_COLUMNS = (
    "home_pre_epa_per_play_r8", "away_pre_epa_per_play_r8",
    "home_pre_epa_per_play_allowed_r8", "away_pre_epa_per_play_allowed_r8",
    "home_pre_success_rate_r8", "away_pre_success_rate_r8",
    "home_pre_success_rate_allowed_r8", "away_pre_success_rate_allowed_r8",
    "home_pre_pass_epa_per_play_r8", "away_pre_pass_epa_per_play_r8",
    "home_pre_pass_epa_per_play_allowed_r8", "away_pre_pass_epa_per_play_allowed_r8",
    "home_pre_rush_epa_per_play_r8", "away_pre_rush_epa_per_play_r8",
    "home_pre_rush_epa_per_play_allowed_r8", "away_pre_rush_epa_per_play_allowed_r8",
    "home_pre_explosive_pass_rate_r8", "away_pre_explosive_pass_rate_r8",
    "home_pre_explosive_pass_rate_allowed_r8", "away_pre_explosive_pass_rate_allowed_r8",
    "home_pre_sack_rate_r8", "away_pre_sack_rate_r8",
    "home_pre_sack_rate_generated_r8", "away_pre_sack_rate_generated_r8",
    "home_pre_interception_rate_r8", "away_pre_interception_rate_r8",
    "home_pre_interception_rate_generated_r8", "away_pre_interception_rate_generated_r8",
    "home_pre_margin_r4", "away_pre_margin_r4", "home_pre_margin_r8", "away_pre_margin_r8",
    "home_pre_qb_epa_per_dropback_r8", "away_pre_qb_epa_per_dropback_r8",
    "home_pre_qb_success_rate_r8", "away_pre_qb_success_rate_r8",
    "home_pre_qb_cpoe_r8", "away_pre_qb_cpoe_r8",
    "home_pre_qb_sack_rate_r8", "away_pre_qb_sack_rate_r8",
    "home_pre_qb_starts", "away_pre_qb_starts",
    "home_starter_change", "away_starter_change", "home_field", "rest_diff",
)


def _require(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = sorted(set(cols) - set(frame.columns))
    if missing:
        raise ValueError("grading source frame missing required columns: " + ", ".join(missing))


def _safe_mean(*series: pd.Series) -> pd.Series:
    return pd.concat(series, axis=1).mean(axis=1, skipna=True)


def _z_to_grade(series: pd.Series, mean: float, std: float, higher_is_better: bool = True) -> pd.Series:
    if not np.isfinite(std) or std <= 1e-12:
        z = pd.Series(0.0, index=series.index)
    else:
        z = (series - mean) / std
    if not higher_is_better:
        z = -z
    return (50.0 + 10.0 * z).clip(20.0, 80.0)


def _fit_reference(frame: pd.DataFrame, raw_cols: list[str]) -> dict[str, tuple[float, float]]:
    reference: dict[str, tuple[float, float]] = {}
    for col in raw_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        reference[col] = (float(values.mean()), float(values.std(ddof=0)))
    return reference


def add_grade_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build transparent pregame-only 0-100 team grades and matchup edges.

    Grades are standardized only from rows earlier than the evaluated season when used
    by walk-forward evaluation. This function itself creates raw grade ingredients;
    `grade_walk_forward` handles chronological reference fitting.
    """
    _require(frame, SOURCE_COLUMNS)
    x = add_dynamic_margin_rating(frame).copy()

    # Raw ingredients. Defense columns are converted to 'higher is better' orientation.
    for side in ("home", "away"):
        p = f"{side}_pre_"
        x[f"raw_{side}_offense"] = _safe_mean(x[p + "epa_per_play_r8"], x[p + "success_rate_r8"])
        x[f"raw_{side}_defense"] = _safe_mean(-x[p + "epa_per_play_allowed_r8"], -x[p + "success_rate_allowed_r8"])
        x[f"raw_{side}_qb"] = _safe_mean(
            x[p + "qb_epa_per_dropback_r8"], x[p + "qb_success_rate_r8"],
            x[p + "qb_cpoe_r8"] / 10.0, -x[p + "qb_sack_rate_r8"],
        )
        x[f"raw_{side}_pass_offense"] = x[p + "pass_epa_per_play_r8"]
        x[f"raw_{side}_rush_offense"] = x[p + "rush_epa_per_play_r8"]
        x[f"raw_{side}_pass_defense"] = -x[p + "pass_epa_per_play_allowed_r8"]
        x[f"raw_{side}_rush_defense"] = -x[p + "rush_epa_per_play_allowed_r8"]
        x[f"raw_{side}_explosiveness"] = _safe_mean(
            x[p + "explosive_pass_rate_r8"], -x[p + "explosive_pass_rate_allowed_r8"]
        )
        x[f"raw_{side}_pressure"] = _safe_mean(x[p + "sack_rate_generated_r8"], -x[p + "sack_rate_r8"])
        x[f"raw_{side}_ball_security"] = _safe_mean(
            -x[p + "interception_rate_r8"], x[p + "interception_rate_generated_r8"]
        )
        x[f"raw_{side}_recent_form"] = _safe_mean(x[p + "margin_r4"], x[p + "margin_r8"])
        starts = np.log1p(pd.to_numeric(x[p + "qb_starts"], errors="coerce").clip(lower=0))
        x[f"raw_{side}_availability"] = starts - 2.0 * pd.to_numeric(x[f"{side}_starter_change"], errors="coerce")

    return x


GRADE_NAMES = (
    "offense", "defense", "qb", "pass_offense", "rush_offense", "pass_defense", "rush_defense",
    "explosiveness", "pressure", "ball_security", "recent_form", "availability",
)


def apply_grade_reference(frame: pd.DataFrame, reference_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply 0-100 grades using a reference fitted only on prior games."""
    x = frame.copy()
    raw_cols = [f"raw_{side}_{name}" for side in ("home", "away") for name in GRADE_NAMES]
    ref = _fit_reference(reference_frame, raw_cols)
    for side in ("home", "away"):
        for name in GRADE_NAMES:
            col = f"raw_{side}_{name}"
            mean, std = ref[col]
            x[f"grade_{side}_{name}"] = _z_to_grade(x[col], mean, std)

    # Composite grades are intentionally simple/transparent; the regression calibrates point value.
    x["grade_home_team"] = _safe_mean(
        x["grade_home_offense"], x["grade_home_defense"], x["grade_home_qb"], x["grade_home_recent_form"]
    )
    x["grade_away_team"] = _safe_mean(
        x["grade_away_offense"], x["grade_away_defense"], x["grade_away_qb"], x["grade_away_recent_form"]
    )
    x["grade_team_edge"] = x["grade_home_team"] - x["grade_away_team"]

    # Matchup edges: offense is compared directly with opponent defense on the same 0-100 scale.
    x["grade_pass_matchup"] = (
        x["grade_home_pass_offense"] - x["grade_away_pass_defense"]
        - (x["grade_away_pass_offense"] - x["grade_home_pass_defense"])
    )
    x["grade_rush_matchup"] = (
        x["grade_home_rush_offense"] - x["grade_away_rush_defense"]
        - (x["grade_away_rush_offense"] - x["grade_home_rush_defense"])
    )
    x["grade_pressure_matchup"] = x["grade_home_pressure"] - x["grade_away_pressure"]
    x["grade_explosive_matchup"] = x["grade_home_explosiveness"] - x["grade_away_explosiveness"]
    x["grade_qb_matchup"] = x["grade_home_qb"] - x["grade_away_qb"]
    x["grade_form_matchup"] = x["grade_home_recent_form"] - x["grade_away_recent_form"]
    x["grade_availability_matchup"] = x["grade_home_availability"] - x["grade_away_availability"]
    x["grade_ball_security_matchup"] = x["grade_home_ball_security"] - x["grade_away_ball_security"]
    return x


MODEL_FEATURES = (
    "v02_dynamic_margin_baseline",
    "grade_team_edge",
    "grade_pass_matchup",
    "grade_rush_matchup",
    "grade_pressure_matchup",
    "grade_explosive_matchup",
    "grade_qb_matchup",
    "grade_form_matchup",
    "grade_availability_matchup",
    "grade_ball_security_matchup",
    "rest_diff",
    "home_field",
)


def _model(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _ats_summary(test: pd.DataFrame, fair: np.ndarray, threshold: float) -> dict:
    market_spread = test["spread_line"].astype(float).map(nflverse_spread_to_market_home_spread)
    edge = fair + market_spread.to_numpy(float)
    side = np.where(edge > 0, "HOME", np.where(edge < 0, "AWAY", "NONE"))
    results = [grade_side(float(m), float(s), str(side_i)) for m, s, side_i in zip(test["home_margin"], market_spread, side)]
    mask = np.abs(edge) >= threshold
    picked = np.asarray(results, dtype=object)[mask]
    wins = int(np.sum(picked == "WIN")); losses = int(np.sum(picked == "LOSS")); pushes = int(np.sum(picked == "PUSH"))
    graded = wins + losses
    net = wins * (100.0 / 110.0) - losses
    return {
        "threshold": float(threshold), "bets": graded, "wins": wins, "losses": losses, "pushes": pushes,
        "win_pct": None if graded == 0 else wins / graded,
        "roi": None if graded == 0 else net / graded,
        "coverage_pct": 100.0 * graded / len(test),
    }


def grade_walk_forward(frame: pd.DataFrame, thresholds=(1.0, 2.0, 3.0, 4.0, 5.0)) -> dict:
    """Evaluate the Walters-style grading architecture.

    2020-2024 select regularization and describe development stability. 2025 is
    explicitly retrospective because it was already observed during earlier model work.
    """
    raw = add_grade_features(frame)
    assert_market_blind_features(MODEL_FEATURES)

    alpha_rows = []
    for alpha in ALPHAS:
        fold_maes = []
        fold_sizes = []
        for season in DEVELOPMENT_SEASONS:
            train_raw = raw[(raw["season"] < season) & raw["home_margin"].notna()].copy()
            test_raw = raw[(raw["season"] == season) & raw["home_margin"].notna()].copy()
            train = apply_grade_reference(train_raw, train_raw)
            test = apply_grade_reference(test_raw, train_raw)
            model = _model(alpha)
            model.fit(train[list(MODEL_FEATURES)], train["home_margin"])
            pred = model.predict(test[list(MODEL_FEATURES)])
            fold_maes.append(float(mean_absolute_error(test["home_margin"], pred)))
            fold_sizes.append(len(test))
        alpha_rows.append({"alpha": alpha, "weighted_mae": float(np.average(fold_maes, weights=fold_sizes))})
    selected_alpha = min(alpha_rows, key=lambda x: (x["weighted_mae"], x["alpha"]))["alpha"]

    folds = []
    for season in (*DEVELOPMENT_SEASONS, RETROSPECTIVE_AUDIT_SEASON):
        train_raw = raw[(raw["season"] < season) & raw["home_margin"].notna()].copy()
        test_raw = raw[(raw["season"] == season) & raw["home_margin"].notna()].copy()
        train = apply_grade_reference(train_raw, train_raw)
        test = apply_grade_reference(test_raw, train_raw)
        model = _model(selected_alpha)
        model.fit(train[list(MODEL_FEATURES)], train["home_margin"])
        fair = model.predict(test[list(MODEL_FEATURES)])
        market_margin = -test["spread_line"].astype(float).map(nflverse_spread_to_market_home_spread).to_numpy(float)
        coefs = model.named_steps["ridge"].coef_[: len(MODEL_FEATURES)]
        folds.append({
            "season": season,
            "role": "development_walk_forward" if season in DEVELOPMENT_SEASONS else "retrospective_audit_not_pristine",
            "games": int(len(test)),
            "beard_mae": float(mean_absolute_error(test["home_margin"], fair)),
            "market_mae": float(mean_absolute_error(test["home_margin"], market_margin)),
            "ats": [_ats_summary(test, fair, t) for t in thresholds],
            "point_weights_standardized": {name: float(value) for name, value in zip(MODEL_FEATURES, coefs)},
        })

    return {
        "label": "BEARD_WALTERS_STYLE_GRADING_V0_3",
        "model_features": list(MODEL_FEATURES),
        "selected_alpha": selected_alpha,
        "alpha_candidates": alpha_rows,
        "folds": folds,
        "scientific_status": "2025 is retrospective, not pristine. Production confirmation requires predeclared 2026 forward predictions.",
        "philosophy": "Visible team grades -> visible matchup grades -> calibrated fair line -> market comparison -> separate qualification.",
    }


def score_game(row: pd.Series, model: Pipeline) -> GradedFairLine:
    fair = float(model.predict(pd.DataFrame([row[list(MODEL_FEATURES)]]))[0])
    team_edge = float(row["grade_team_edge"])
    matchup_keys = (
        "grade_pass_matchup", "grade_rush_matchup", "grade_pressure_matchup", "grade_explosive_matchup",
        "grade_qb_matchup", "grade_form_matchup", "grade_availability_matchup", "grade_ball_security_matchup",
    )
    matchup_edge = float(np.nanmean([row[k] for k in matchup_keys]))
    components = {k: float(row[k]) if pd.notna(row[k]) else float("nan") for k in MODEL_FEATURES}
    return GradedFairLine(
        home_team=str(row["home_team"]), away_team=str(row["away_team"]),
        home_team_grade=float(row["grade_home_team"]), away_team_grade=float(row["grade_away_team"]),
        base_grade_edge=team_edge, matchup_grade_edge=matchup_edge,
        fair_home_margin=fair, projected_home_spread=-fair, components=components,
    )
