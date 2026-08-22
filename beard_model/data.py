from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

from .config import DEFAULT_CONFIG, ModelConfig


SCHEDULE_REQUIRED = {
    "game_id", "season", "game_type", "week", "gameday", "away_team", "home_team",
    "away_score", "home_score", "away_rest", "home_rest", "spread_line", "total_line",
    "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name", "location",
}
PBP_REQUIRED = {
    "game_id", "season", "week", "posteam", "defteam", "play_type", "epa", "success",
    "yards_gained", "down", "sack", "interception",
}


@dataclass(frozen=True)
class SourceMetadata:
    path: str
    sha256: str
    rows: int
    columns: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _download(url: str, destination: Path, timeout: float = 120.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    tmp.replace(destination)
    return destination


def load_schedules(
    cache_dir: str | Path = "data/cache",
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
) -> tuple[pd.DataFrame, SourceMetadata]:
    path = Path(cache_dir) / "games.csv"
    if refresh or not path.exists():
        _download(config.schedule_url, path)
    df = pd.read_csv(path, low_memory=False)
    _validate_columns(df, SCHEDULE_REQUIRED, "schedule dataset")
    allowed_types = {"REG", "WC", "DIV", "CON", "SB"} if config.include_postseason else {"REG"}
    df = df[df["game_type"].isin(allowed_types)].copy()
    df["gameday"] = pd.to_datetime(df["gameday"], errors="raise")
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    meta = SourceMetadata(str(path), sha256_file(path), len(df), tuple(df.columns))
    return df, meta


def load_pbp(
    season: int,
    cache_dir: str | Path = "data/cache",
    config: ModelConfig = DEFAULT_CONFIG,
    refresh: bool = False,
) -> tuple[pd.DataFrame, SourceMetadata]:
    path = Path(cache_dir) / f"play_by_play_{season}.parquet"
    if refresh or not path.exists():
        _download(config.pbp_url_template.format(season=season), path)
    df = pd.read_parquet(path)
    _validate_columns(df, PBP_REQUIRED, f"PBP {season}")
    if "season_type" in df.columns:
        allowed_types = {"REG", "POST"} if config.include_postseason else {"REG"}
        df = df[df["season_type"].isin(allowed_types)].copy()
    df = df[df["season"] == season].copy()
    meta = SourceMetadata(str(path), sha256_file(path), len(df), tuple(df.columns))
    return df, meta


def available_optional_columns(df: pd.DataFrame) -> dict[str, bool]:
    optional = [
        "pass", "rush", "qb_dropback", "qb_epa", "cpoe", "passer_player_id",
        "passer_player_name", "passer_id", "passer", "complete_pass", "air_yards",
    ]
    return {name: name in df.columns for name in optional}
