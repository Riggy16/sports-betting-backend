# BEARD Model / Sports Betting Backend

This repository is the NFL modeling/backend codebase for **BEARD Analytics**, under **Bearded Edge Sports**.

- **B** — Baseline Team Strength
- **E** — Efficiency & Explosiveness
- **A** — Availability & Adjustments
- **R** — Rest / Road / Conditions
- **D** — Direct Matchup

## Non-negotiable integrity rule

The fair-line model is **market blind**. Sportsbook spreads, totals, moneylines, prices, vig/juice, line movement, CLV, and derivatives of those fields are forbidden in the feature matrix used to fit or generate the BEARD fair line. Market data is joined only **after** a fair prediction is generated and frozen.

## Chronological split

- Training/development: **2015–2023**
- Validation + threshold selection: **2024**
- Untouched holdout: **2025**

If 2025 is used to tune features, hyperparameters, blend weights, or qualification thresholds after its results are viewed, it stops being an untouched holdout.

## Historical sources

- Schedules/results: `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
- Play-by-play: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet`

Follow the applicable nflverse/nflfastR license and attribution terms.

## Critical spread convention

nflverse `spread_line` uses **positive = home team favored**. Standard sportsbook display uses a negative number for a favorite. BEARD therefore converts it **after fair predictions are frozen**:

`market_home_spread = -nflverse_spread_line`

Any ATS test using the schedule field is labeled **CLOSING-LINE BACKTEST**. It is not an early-line simulation and it is not a CLV study.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m beard_model.pipeline --help
python -m beard_model.backtest --help
```

## Full historical run

```bash
python -m beard_model.pipeline --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 --output artifacts/beard_matchups.parquet
python -m beard_model.backtest --input artifacts/beard_matchups.parquet --output artifacts/backtest.json
```

The pipeline constructs team/QB/matchup features first and only afterward attaches nflverse `spread_line` for the explicitly labeled closing-line backtest.

## Current verified status

- **31/31** local unit/synthetic regression tests passing.
- Covered: market-feature rejection, nflverse spread-sign normalization, same-game rolling leakage, future-game feature construction, QB pregame shifting, holdout isolation, fair-line contracts, direct matchup construction, and market/fair-frame separation.
- Unit/synthetic tests validate **code behavior**, not betting profitability.
- The full 2015–2025 real-data chronological backtest has **not yet produced an accepted performance result**.
- Production qualification thresholds remain unvalidated until the validation/holdout run is complete.
