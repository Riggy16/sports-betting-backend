# BEARD model → Supabase mapping

The production schema deliberately separates market-blind football predictions from sportsbook observations.

| Model concept | Supabase table | Integrity rule |
|---|---|---|
| Source audit metadata | `source_ingest_runs` | Store source/hash/status; never silently substitute data. |
| Game truth/context | `nfl_games` | Schedule, scores, venue, rest, historical starters. |
| Timestamped sportsbook observations | `market_lines` | **Post-prediction only.** |
| Pregame team snapshots | `team_week_features` | Every feature must be computable before kickoff. |
| Pregame QB snapshots | `qb_week_features` | Starter statistics are shifted; availability is separately sourced. |
| Reproducible model metadata | `model_versions` | Split, feature version, code/data hashes, metrics. |
| Frozen fair line | `fair_predictions` | **Market blind. No spread/odds/moneyline inputs.** |
| Fair line vs market | `edge_evaluations` | Post-market qualification and confidence live here. |
| Permanent grading | `prediction_results` | Retain wins, losses, pushes and genuine CLV only when timestamps support it. |
| Published team strength | `power_ratings` | Model-versioned weekly ratings. |

`FairPrediction` maps to `fair_predictions`; `MarketLineSnapshot` maps to `market_lines`; `EdgeEvaluation` maps to `edge_evaluations`; `ModelMetrics` belongs in `model_versions.metrics` and/or immutable backtest artifacts.
