# BEARD Model / Sports Betting Backend

This repository is the backend/model codebase for **BEARD Analytics**, the NFL analytics product under **Bearded Edge Sports**.

BEARD stands for:

- **B** — Baseline Team Strength
- **E** — Efficiency & Explosiveness
- **A** — Availability & Adjustments
- **R** — Rest / Road / Conditions
- **D** — Direct Matchup

## Non-negotiable model-integrity rule

The BEARD fair-line model is **market blind**. Sportsbook spreads, totals, moneylines, prices, vig/juice, line movement, CLV, and derivatives of those fields are forbidden from the feature matrix used to fit or generate a fair line. Market data is joined only **after** a fair prediction is generated and frozen.

This separation is enforced in code and mirrors the BEARD Analytics database architecture: fair predictions are stored separately from market snapshots and post-market edge evaluations.

## Chronological split

Default v0.1 development protocol:

- Training/development: **2015–2023**
- Validation + qualification-threshold selection: **2024**
- Untouched holdout: **2025**

Do not tune features, model hyperparameters, blend weights, or qualification thresholds on 2025 after viewing holdout performance. If that happens, 2025 is no longer an untouched holdout.

## Historical sources

- Schedules/results: `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
- Play-by-play: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet`

The model is designed around public nflverse/nflfastR data. nflfastR documents play-by-play back to 1999 and derived fields including EPA, success, QB EPA, and CPOE where available. Follow the upstream nflverse repository/data licenses and attribution requirements.

## Backtest labeling

The nflverse schedule file exposes a historical `spread_line` field but does not provide the complete timestamped line path for every historical game. Any ATS evaluation based solely on that field must be labeled **CLOSING-LINE BACKTEST**. It is not an early-line simulation and it is not a CLV study.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m beard_model.backtest --help
```

## Data flow

1. Acquire and hash schedule/PBP source data.
2. Aggregate PBP into team-game and QB-game metrics.
3. Shift/roll metrics so every model row contains **pregame-only** information.
4. Build direct matchup features.
5. Fit Ridge + gradient-boosting fair-line models only on training data; use validation for selection/tuning.
6. Freeze fair predictions.
7. Join sportsbook market data afterward.
8. Select/freeze qualification policy on validation only.
9. Evaluate 2025 once as untouched holdout.

## Current status

The v0.1 source package has passed **27/27 local unit/synthetic integrity tests**, including market-leakage rejection, same-game rolling-feature leakage, QB pregame shifting, spread-sign grading, and holdout isolation. **No claim of real betting performance is made from those tests.** The first full 2015–2025 nflverse play-by-play backtest is still required before any model-performance claim or production qualification thresholds are published.
