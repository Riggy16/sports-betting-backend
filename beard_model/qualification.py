from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import sqrt

import numpy as np
import pandas as pd

from .config import CandidateThresholds, DEFAULT_CONFIG, ModelConfig
from .contracts import EdgeEvaluation, FairPrediction, MarketLineSnapshot
from .splits import threshold_selection_seasons


@dataclass(frozen=True)
class QualificationPolicy:
    thresholds: CandidateThresholds
    qualification_version: str = "qv1-unvalidated"
    selected_on_seasons: tuple[int, ...] = ()
    validated: bool = False


def nflverse_spread_to_market_home_spread(spread_line: float) -> float:
    """Convert nflverse spread_line to standard sportsbook home spread."""
    return -float(spread_line)


def home_cover_edge(home_fair_margin: float, market_home_spread: float) -> float:
    """Positive = value on HOME; negative = value on AWAY."""
    return float(home_fair_margin) + float(market_home_spread)


def ats_home_result(actual_home_margin: float, market_home_spread: float) -> int:
    covered = float(actual_home_margin) + float(market_home_spread)
    return 1 if covered > 0 else (-1 if covered < 0 else 0)


def grade_side(actual_home_margin: float, market_home_spread: float, side: str) -> str:
    home_result = ats_home_result(actual_home_margin, market_home_spread)
    if home_result == 0:
        return "PUSH"
    if side == "HOME":
        return "WIN" if home_result > 0 else "LOSS"
    if side == "AWAY":
        return "WIN" if home_result < 0 else "LOSS"
    return "VOID"


def _confidence_score(edge_abs: float, uncertainty: float, disagreement: float, data_quality: float, qb_conf: float | None) -> float:
    qb = 0.5 if qb_conf is None or np.isnan(qb_conf) else float(np.clip(qb_conf, 0, 1))
    score = (
        35.0
        + min(edge_abs, 5.0) * 7.0
        + (float(np.clip(data_quality, 0, 100)) - 70.0) * 0.25
        + (qb - 0.5) * 20.0
        - max(uncertainty - 10.0, 0.0) * 1.5
        - disagreement * 2.0
    )
    return float(np.clip(score, 0.0, 100.0))


def evaluate_edge(
    fair: FairPrediction,
    market: MarketLineSnapshot,
    policy: QualificationPolicy | None = None,
    qb_availability_confidence: float | None = None,
) -> EdgeEvaluation:
    if fair.game_id != market.game_id:
        raise ValueError("Fair prediction and market snapshot must reference the same game.")
    if market.home_spread is None:
        return EdgeEvaluation(
            game_id=fair.game_id, evaluated_at=datetime.now(timezone.utc), recommended_side="NONE", edge_points=0.0,
            confidence_score=0.0, qualification_status="NO_PLAY",
            qualification_version=(policy.qualification_version if policy else "qv1-unvalidated"),
            top_concerns=["No market spread available"],
        )
    policy = policy or QualificationPolicy(DEFAULT_CONFIG.candidate_thresholds)
    t = policy.thresholds
    edge = home_cover_edge(fair.home_fair_margin, market.home_spread)
    edge_abs = abs(edge)
    side = "HOME" if edge > 0 else ("AWAY" if edge < 0 else "NONE")
    qb_known = qb_availability_confidence is not None and not np.isnan(qb_availability_confidence)
    qb = float(qb_availability_confidence) if qb_known else 0.0
    u = fair.prediction_stddev
    d = fair.ensemble_disagreement
    q = fair.data_quality_score

    status = "NO_PLAY"
    if edge_abs >= t.lean_edge:
        status = "LEAN"
    if (
        edge_abs >= t.qualified_edge and u <= t.max_uncertainty_qualified and d <= t.max_disagreement_qualified
        and q >= t.min_data_quality_qualified and qb_known and qb >= t.min_qb_availability_qualified
    ):
        status = "QUALIFIED"
    if (
        edge_abs >= t.strong_edge and u <= t.max_uncertainty_strong and d <= t.max_disagreement_strong
        and q >= t.min_data_quality_strong and qb_known and qb >= t.min_qb_availability_strong
    ):
        status = "STRONG"
    if side == "NONE":
        status = "NO_PLAY"

    concerns = []
    if not qb_known:
        concerns.append("QB availability confidence unavailable; cannot qualify above LEAN")
    if u > t.max_uncertainty_qualified:
        concerns.append("Prediction uncertainty is high")
    if d > t.max_disagreement_qualified:
        concerns.append("Ensemble models disagree")
    if q < t.min_data_quality_qualified:
        concerns.append("Data completeness is below qualification standard")
    advantages = [f"Fair line differs from market by {edge_abs:.1f} points"] if edge_abs else []
    return EdgeEvaluation(
        game_id=fair.game_id,
        evaluated_at=datetime.now(timezone.utc),
        recommended_side=side,
        edge_points=edge_abs,
        confidence_score=_confidence_score(edge_abs, u, d, q, qb_availability_confidence),
        qualification_status=status,
        qualification_version=policy.qualification_version,
        top_advantages=advantages,
        top_concerns=concerns,
    )


def _wilson_lower_bound(wins: int, losses: int, z: float = 1.0) -> float:
    n = wins + losses
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def select_edge_thresholds_from_validation(
    validation: pd.DataFrame,
    config: ModelConfig = DEFAULT_CONFIG,
    min_bets: int = 25,
) -> QualificationPolicy:
    """Select edge cutoffs on validation only; holdout rows are rejected."""
    required = {"season", "edge_abs", "ats_result"}
    missing = sorted(required - set(validation.columns))
    if missing:
        raise ValueError(f"Validation threshold frame missing: {', '.join(missing)}")
    allowed = set(threshold_selection_seasons(config))
    seasons = set(int(s) for s in validation["season"].dropna().unique())
    if not seasons <= allowed:
        raise ValueError(f"Threshold selection may use validation seasons only {sorted(allowed)}; got {sorted(seasons)}")

    best: tuple[float, float, int] | None = None
    for threshold in np.arange(1.0, 4.01, 0.25):
        sub = validation[validation["edge_abs"] >= threshold]
        wins = int((sub["ats_result"] == "WIN").sum())
        losses = int((sub["ats_result"] == "LOSS").sum())
        n = wins + losses
        if n < min_bets:
            continue
        score = _wilson_lower_bound(wins, losses, z=1.0)
        candidate = (score, float(threshold), n)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[2] > best[2]):
            best = candidate

    base = config.candidate_thresholds
    if best is None:
        return QualificationPolicy(base, qualification_version="qv1-validation-insufficient", selected_on_seasons=tuple(sorted(allowed)), validated=False)
    _, qualified, _ = best
    strong = max(qualified + 1.0, base.strong_edge)
    selected = replace(base, qualified_edge=qualified, strong_edge=strong, label="VALIDATION_SELECTED_EDGE_CUTOFFS")
    return QualificationPolicy(selected, qualification_version="qv1-frozen-before-holdout", selected_on_seasons=tuple(sorted(allowed)), validated=True)
