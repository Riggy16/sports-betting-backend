from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class FairPrediction(BaseModel):
    """Frozen market-blind model output. Intentionally contains no sportsbook fields."""

    model_config = ConfigDict(extra="forbid")

    game_id: str
    model_version: str
    generated_at: datetime
    data_as_of: datetime
    home_fair_margin: float
    fair_total: float | None = None
    projected_home_score: float | None = None
    projected_away_score: float | None = None
    home_win_probability: float = Field(ge=0.0, le=1.0)
    prediction_stddev: float = Field(gt=0.0)
    ensemble_disagreement: float = Field(ge=0.0)
    data_quality_score: float = Field(ge=0.0, le=100.0)
    feature_snapshot_hash: str


class MarketLineSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    observed_at: datetime
    sportsbook: str = "consensus"
    home_spread: float | None = None
    total: float | None = None
    home_spread_price: int | None = None
    away_spread_price: int | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    line_type: Literal["open", "snapshot", "close"] = "snapshot"


class EdgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    fair_prediction_id: str | None = None
    market_line_id: str | None = None
    evaluated_at: datetime
    recommended_side: Literal["HOME", "AWAY", "NONE"]
    edge_points: float
    confidence_score: float = Field(ge=0.0, le=100.0)
    qualification_status: Literal["NO_PLAY", "LEAN", "QUALIFIED", "STRONG"]
    qualification_version: str = "qv1-unvalidated"
    top_advantages: list[str] = Field(default_factory=list)
    top_concerns: list[str] = Field(default_factory=list)
    public_explanation: str | None = None


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_name: str
    games: int
    margin_mae: float | None = None
    margin_rmse: float | None = None
    total_mae: float | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    ats_wins: int = 0
    ats_losses: int = 0
    ats_pushes: int = 0
    ats_win_pct: float | None = None
    roi: float | None = None
    coverage_pct: float | None = None
