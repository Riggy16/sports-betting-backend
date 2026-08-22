from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateThresholds:
    """UNVALIDATED candidate thresholds. Validation data must select/freeze final values."""

    lean_edge: float = 1.0
    qualified_edge: float = 2.0
    strong_edge: float = 3.0
    max_uncertainty_qualified: float = 15.0
    max_uncertainty_strong: float = 13.0
    max_disagreement_qualified: float = 4.0
    max_disagreement_strong: float = 3.0
    min_data_quality_qualified: float = 75.0
    min_data_quality_strong: float = 85.0
    min_qb_availability_qualified: float = 0.75
    min_qb_availability_strong: float = 0.90
    label: str = "CANDIDATE_UNVALIDATED"


@dataclass(frozen=True)
class ModelConfig:
    training_seasons: tuple[int, ...] = tuple(range(2015, 2024))
    validation_seasons: tuple[int, ...] = (2024,)
    holdout_seasons: tuple[int, ...] = (2025,)
    random_seed: int = 42
    feature_version: str = "v1"
    schedule_url: str = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    pbp_url_template: str = (
        "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
        "play_by_play_{season}.parquet"
    )
    include_postseason: bool = False
    candidate_thresholds: CandidateThresholds = field(default_factory=CandidateThresholds)

    def __post_init__(self) -> None:
        train, val, hold = map(set, (self.training_seasons, self.validation_seasons, self.holdout_seasons))
        if train & val or train & hold or val & hold:
            raise ValueError("Training, validation, and holdout seasons must be disjoint.")
        if not self.validation_seasons:
            raise ValueError("At least one validation season is required.")
        if not self.holdout_seasons:
            raise ValueError("At least one untouched holdout season is required.")


DEFAULT_CONFIG = ModelConfig()
