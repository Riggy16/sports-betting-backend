import numpy as np
import pandas as pd

from beard_model.diagnostics import edge_bucket_table, market_margin_metrics, feature_health


def test_market_baseline_uses_nflverse_margin_convention_directly():
    actual = pd.DataFrame({"home_margin":[7.0, -3.0], "spread_line":[4.0, -1.0]})
    pred = pd.DataFrame({"home_fair_margin":[5.0, 1.0]})
    m = market_margin_metrics(actual, pred)
    assert m["market_margin_mae"] == 2.5
    assert m["beard_margin_mae"] == 3.0
    assert m["beard_mae_minus_market"] == 0.5


def test_edge_buckets_are_non_overlapping():
    rows = pd.DataFrame({
        "edge_abs":[0.5,1.5,2.5,3.5,5.0,7.0],
        "ats_result":["WIN","LOSS","WIN","LOSS","WIN","LOSS"],
    })
    table = edge_bucket_table(rows, bins=(0,1,2,3,4,6,float('inf')))
    assert sum(x["games"] for x in table) == 6
    assert table[0]["wins"] == 1
    assert table[-1]["losses"] == 1


def test_feature_health_detects_dead_feature_without_market_leakage():
    frame = pd.DataFrame({
        "season":[2024,2025], "week":[1,1], "game_id":["a","b"],
        "gameday":pd.to_datetime(["2024-09-01","2025-09-01"]),
        "home_team":["A","A"], "away_team":["B","B"],
        "home_margin":[1.0,2.0], "total_points":[40.0,41.0],
        "spread_line":[1.0,1.5],
        "live_feature":[1.0,2.0], "dead_feature":[np.nan,np.nan],
    })
    h = feature_health(frame)
    assert "dead_feature" in h["all_missing"]
