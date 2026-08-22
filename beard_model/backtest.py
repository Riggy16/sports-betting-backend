from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from .config import DEFAULT_CONFIG, ModelConfig
from .fair_line import FairLineEnsemble
from .matchup import model_feature_columns
from .qualification import grade_side, nflverse_spread_to_market_home_spread, select_edge_thresholds_from_validation
from .splits import split_frame


def _prediction_metrics(actual: pd.DataFrame, pred: pd.DataFrame, name: str) -> dict:
    y = actual["home_margin"].astype(float).to_numpy()
    p = pred["home_fair_margin"].to_numpy()
    actual_home_win = (y > 0).astype(int)
    prob = np.clip(pred["home_win_probability"].to_numpy(), 1e-6, 1 - 1e-6)
    result = {
        "split": name,
        "games": len(actual),
        "margin_mae": float(mean_absolute_error(y, p)),
        "margin_rmse": float(np.sqrt(mean_squared_error(y, p))),
        "brier_score": float(brier_score_loss(actual_home_win, prob)),
        "log_loss": float(log_loss(actual_home_win, prob, labels=[0, 1])),
    }
    if "total_points" in actual.columns and pred["fair_total"].notna().all():
        result["total_mae"] = float(mean_absolute_error(actual["total_points"], pred["fair_total"]))
    return result


def _closing_line_rows(actual: pd.DataFrame, pred: pd.DataFrame, spread_col: str = "spread_line") -> pd.DataFrame:
    """Join the nflverse historical closing line only after fair predictions are frozen."""
    if spread_col not in actual.columns:
        raise ValueError(f"Closing-line backtest requires `{spread_col}` in a separate post-prediction join.")
    rows = pd.DataFrame(index=actual.index)
    rows["season"] = actual["season"].astype(int)
    rows["home_margin"] = actual["home_margin"].astype(float)
    rows["nflverse_spread_line"] = actual[spread_col].astype(float)
    rows["market_home_spread"] = rows["nflverse_spread_line"].map(nflverse_spread_to_market_home_spread)
    rows["home_fair_margin"] = pred["home_fair_margin"].astype(float)
    rows["edge_signed_home"] = rows["home_fair_margin"] + rows["market_home_spread"]
    rows["edge_abs"] = rows["edge_signed_home"].abs()
    rows["recommended_side"] = np.where(rows["edge_signed_home"] > 0, "HOME", np.where(rows["edge_signed_home"] < 0, "AWAY", "NONE"))
    rows["ats_result"] = [
        grade_side(m, s, side) for m, s, side in zip(rows["home_margin"], rows["market_home_spread"], rows["recommended_side"])
    ]
    return rows


def _ats_summary(rows: pd.DataFrame, threshold: float) -> dict:
    sub = rows[rows["edge_abs"] >= threshold].copy()
    wins = int((sub["ats_result"] == "WIN").sum())
    losses = int((sub["ats_result"] == "LOSS").sum())
    pushes = int((sub["ats_result"] == "PUSH").sum())
    decisions = wins + losses
    net_units = wins * (100 / 110) - losses
    return {
        "threshold": threshold,
        "bets": decisions,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_pct": None if decisions == 0 else wins / decisions,
        "roi": None if decisions == 0 else net_units / decisions,
        "coverage_pct": 0.0 if len(rows) == 0 else 100.0 * decisions / len(rows),
    }


def run_closing_line_backtest(frame: pd.DataFrame, config: ModelConfig = DEFAULT_CONFIG) -> dict:
    """Strict chronological fair-line test, then post-prediction closing-spread evaluation."""
    splits = split_frame(frame, config)
    train, val, hold = splits["train"], splits["validation"], splits["holdout"]
    if min(len(train), len(val), len(hold)) == 0:
        raise ValueError("Train, validation, and holdout splits must all contain games.")

    if "spread_line" not in frame.columns:
        raise ValueError("CLOSING-LINE BACKTEST requires spread_line as the explicit post-market field.")
    fair_source = frame.drop(columns=["spread_line"])
    features = model_feature_columns(fair_source)
    model = FairLineEnsemble(config.random_seed).fit(train, val, features)
    val_pred = model.predict_frame(val)
    val_metrics = _prediction_metrics(val, val_pred, "validation")
    val_close = _closing_line_rows(val, val_pred)
    policy = select_edge_thresholds_from_validation(val_close[["season", "edge_abs", "ats_result"]], config)

    model.refit_on_train_plus_validation(train, val)
    hold_pred = model.predict_frame(hold)
    hold_metrics = _prediction_metrics(hold, hold_pred, "holdout")
    hold_close = _closing_line_rows(hold, hold_pred)

    q = policy.thresholds.qualified_edge
    s = policy.thresholds.strong_edge
    return {
        "label": "CLOSING-LINE BACKTEST",
        "warning": "nflverse spread_line is sign-normalized to standard sportsbook notation after fair predictions are frozen; this is not an early-line or CLV simulation.",
        "model_metadata_hash": model.metadata_hash(),
        "features": features,
        "validation_metrics": val_metrics,
        "qualification_policy": {
            "version": policy.qualification_version,
            "validated": policy.validated,
            "selected_on_seasons": list(policy.selected_on_seasons),
            "qualified_edge": q,
            "strong_edge": s,
        },
        "validation_ats": {"qualified": _ats_summary(val_close, q), "strong": _ats_summary(val_close, s)},
        "holdout_metrics": hold_metrics,
        "holdout_ats": {"qualified": _ats_summary(hold_close, q), "strong": _ats_summary(hold_close, s)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BEARD strict chronological closing-line backtest")
    parser.add_argument("--input", help="Parquet/CSV matchup feature frame containing outcomes and spread_line")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    if not args.input:
        parser.print_help()
        return
    frame = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    result = run_closing_line_backtest(frame)
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
