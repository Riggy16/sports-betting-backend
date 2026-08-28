from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .backtest import _ats_summary, _closing_line_rows
from .config import ModelConfig
from .fair_line import FairLineEnsemble
from .matchup import model_feature_columns
from .splits import split_frame


DEFAULT_FIXED_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_EDGE_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, float("inf"))


def _fair_feature_columns(frame: pd.DataFrame) -> list[str]:
    if "spread_line" not in frame.columns:
        raise ValueError("Diagnostics require nflverse spread_line for market comparison.")
    return model_feature_columns(frame.drop(columns=["spread_line"]))


def market_margin_metrics(actual: pd.DataFrame, fair_pred: pd.DataFrame) -> dict:
    required = {"home_margin", "spread_line"}
    missing = sorted(required - set(actual.columns))
    if missing:
        raise ValueError(f"Missing market baseline columns: {', '.join(missing)}")
    y = actual["home_margin"].astype(float).to_numpy()
    market = actual["spread_line"].astype(float).to_numpy()
    beard = fair_pred["home_fair_margin"].astype(float).to_numpy()
    return {
        "games": int(len(actual)),
        "beard_margin_mae": float(mean_absolute_error(y, beard)),
        "market_margin_mae": float(mean_absolute_error(y, market)),
        "beard_margin_rmse": float(np.sqrt(mean_squared_error(y, beard))),
        "market_margin_rmse": float(np.sqrt(mean_squared_error(y, market))),
        "beard_mae_minus_market": float(mean_absolute_error(y, beard) - mean_absolute_error(y, market)),
    }


def edge_threshold_table(rows: pd.DataFrame, thresholds: Iterable[float] = DEFAULT_FIXED_THRESHOLDS) -> list[dict]:
    return [_ats_summary(rows, float(t)) for t in thresholds]


def edge_bucket_table(rows: pd.DataFrame, bins: Iterable[float] = DEFAULT_EDGE_BINS) -> list[dict]:
    edges = list(float(x) for x in bins)
    if len(edges) < 2:
        raise ValueError("At least two edge-bin boundaries are required.")
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = rows[(rows["edge_abs"] >= lo) & (rows["edge_abs"] < hi)].copy()
        wins = int((sub["ats_result"] == "WIN").sum())
        losses = int((sub["ats_result"] == "LOSS").sum())
        pushes = int((sub["ats_result"] == "PUSH").sum())
        decisions = wins + losses
        net = wins * (100 / 110) - losses
        out.append({
            "edge_min": lo,
            "edge_max": None if np.isinf(hi) else hi,
            "games": int(len(sub)),
            "bets": decisions,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": None if decisions == 0 else wins / decisions,
            "roi": None if decisions == 0 else net / decisions,
        })
    return out


def feature_health(frame: pd.DataFrame) -> dict:
    features = _fair_feature_columns(frame)
    all_missing = [c for c in features if frame[c].isna().all()]
    near_constant = []
    for c in features:
        nonnull = frame[c].dropna()
        if len(nonnull) and nonnull.nunique() <= 1:
            near_constant.append(c)
    duplicate_pairs = []
    seen: dict[int, str] = {}
    for c in features:
        h = int(pd.util.hash_pandas_object(frame[c], index=False).sum())
        if h in seen and frame[c].equals(frame[seen[h]]):
            duplicate_pairs.append([seen[h], c])
        else:
            seen[h] = c
    return {
        "feature_count": len(features),
        "all_missing": all_missing,
        "near_constant": near_constant,
        "exact_duplicate_pairs": duplicate_pairs,
    }


def one_fold_diagnostics(
    frame: pd.DataFrame,
    test_season: int,
    first_training_season: int = 2015,
    fixed_thresholds: Iterable[float] = DEFAULT_FIXED_THRESHOLDS,
    random_seed: int = 42,
) -> dict:
    validation_season = int(test_season) - 1
    training_seasons = tuple(range(int(first_training_season), validation_season))
    if not training_seasons:
        raise ValueError("Walk-forward fold needs at least one training season.")
    config = ModelConfig(
        training_seasons=training_seasons,
        validation_seasons=(validation_season,),
        holdout_seasons=(int(test_season),),
        random_seed=random_seed,
    )
    splits = split_frame(frame, config)
    train, val, test = splits["train"], splits["validation"], splits["holdout"]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"Incomplete walk-forward split for {test_season}.")
    features = _fair_feature_columns(frame)
    model = FairLineEnsemble(random_seed).fit(train, val, features)
    model.refit_on_train_plus_validation(train, val)
    pred = model.predict_frame(test)
    rows = _closing_line_rows(test, pred)
    return {
        "test_season": int(test_season),
        "train_seasons": [int(x) for x in training_seasons],
        "validation_season": validation_season,
        "games": int(len(test)),
        "margin_comparison": market_margin_metrics(test, pred),
        "fixed_edge_thresholds": edge_threshold_table(rows, fixed_thresholds),
        "edge_buckets": edge_bucket_table(rows),
    }


def walk_forward_diagnostics(
    frame: pd.DataFrame,
    test_seasons: Iterable[int] = tuple(range(2020, 2026)),
    first_training_season: int = 2015,
    fixed_thresholds: Iterable[float] = DEFAULT_FIXED_THRESHOLDS,
    random_seed: int = 42,
) -> dict:
    folds = [
        one_fold_diagnostics(frame, int(season), first_training_season, fixed_thresholds, random_seed)
        for season in test_seasons
    ]
    aggregate = []
    for threshold in fixed_thresholds:
        wins = losses = pushes = bets = games = 0
        for fold in folds:
            row = next(x for x in fold["fixed_edge_thresholds"] if x["threshold"] == float(threshold))
            wins += row["wins"]
            losses += row["losses"]
            pushes += row["pushes"]
            bets += row["bets"]
            games += fold["games"]
        net = wins * (100 / 110) - losses
        aggregate.append({
            "threshold": float(threshold),
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": None if bets == 0 else wins / bets,
            "roi": None if bets == 0 else net / bets,
            "coverage_pct": None if games == 0 else 100.0 * bets / games,
        })
    return {
        "label": "WALK_FORWARD_CLOSING_LINE_DIAGNOSTIC",
        "note": "Each test season is predicted after training on earlier seasons and validating on the immediately prior season. Fixed edge thresholds are reported without optimizing to the test season.",
        "feature_health": feature_health(frame),
        "folds": folds,
        "aggregate_fixed_thresholds": aggregate,
    }
