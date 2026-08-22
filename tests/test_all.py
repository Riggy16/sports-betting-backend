import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from beard_model.backtest import run_closing_line_backtest, _closing_line_rows
from beard_model.config import ModelConfig, DEFAULT_CONFIG
from beard_model.contracts import FairPrediction, EdgeEvaluation, MarketLineSnapshot
from beard_model.fair_line import FairLineEnsemble
from beard_model.features import add_pregame_rolling_features, team_game_metrics, schedule_team_game_metrics
from beard_model.integrity import assert_market_blind_features
from beard_model.matchup import build_matchup_frame, model_feature_columns
from beard_model.pipeline import build_feature_frame_from_frames, build_historical_backtest_frame
from beard_model.qb import qb_game_metrics, build_pregame_qb_features
from beard_model.qualification import evaluate_edge, grade_side, home_cover_edge, nflverse_spread_to_market_home_spread, select_edge_thresholds_from_validation
from beard_model.splits import threshold_selection_seasons


def test_synthetic_backtest_keeps_holdout_out_of_threshold_selection():
    rng = np.random.default_rng(42)
    rows = []
    for season, n in [(2022,60),(2023,60),(2024,60),(2025,60)]:
        for i in range(n):
            f1, f2 = rng.normal(), rng.normal()
            margin = 3.0*f1 - 1.5*f2 + rng.normal(scale=7)
            total = 44 + 2*f2 + rng.normal(scale=8)
            spread = 0.7*(3.0*f1 - 1.5*f2) + rng.normal(scale=2)
            rows.append({"game_id":f"{season}_{i}","season":season,"week":i%18+1,"gameday":f"{season}-09-{i%28+1:02d}","home_team":"H","away_team":"A","feature_one":f1,"feature_two":f2,"home_margin":margin,"total_points":total,"spread_line":spread})
    df = pd.DataFrame(rows)
    cfg = ModelConfig(training_seasons=(2022,2023), validation_seasons=(2024,), holdout_seasons=(2025,))
    result = run_closing_line_backtest(df, cfg)
    assert result["label"] == "CLOSING-LINE BACKTEST"
    assert result["qualification_policy"]["selected_on_seasons"] == [2024]
    assert "spread_line" not in result["features"]
    assert result["holdout_metrics"]["split"] == "holdout"


def test_closing_line_sign_matches_nflverse_documented_convention():
    actual = pd.DataFrame({"season":[2023], "home_margin":[-1.0], "spread_line":[4.5]})
    pred = pd.DataFrame({"home_fair_margin":[2.0]})
    rows = _closing_line_rows(actual, pred)
    assert rows.iloc[0]["market_home_spread"] == -4.5
    assert rows.iloc[0]["recommended_side"] == "AWAY"


def test_fair_prediction_contains_no_market_fields():
    fields = set(FairPrediction.model_fields)
    forbidden_fragments = ("spread", "odds", "moneyline", "market", "sportsbook", "clv")
    assert not [f for f in fields if any(t in f.lower() for t in forbidden_fragments)]


def test_confidence_is_post_market_and_not_win_probability():
    assert "home_win_probability" in FairPrediction.model_fields
    assert "confidence_score" not in FairPrediction.model_fields
    assert "confidence_score" in EdgeEvaluation.model_fields
    assert "home_win_probability" not in EdgeEvaluation.model_fields


def synthetic_frame(season, n=50, seed=1):
    rng = np.random.default_rng(seed + season)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({"game_id":[f"{season}_{i}" for i in range(n)], "season":[season]*n, "week":[i%18+1 for i in range(n)], "gameday":pd.date_range(f"{season}-09-01", periods=n, freq="D"), "home_team":["H"]*n, "away_team":["A"]*n, "feature_one":x1, "feature_two":x2, "home_margin":4*x1 - 2*x2 + rng.normal(scale=5,size=n), "total_points":44 + 3*x2 + rng.normal(scale=7,size=n)})


def test_fair_line_rejects_market_feature():
    tr, va = synthetic_frame(2023), synthetic_frame(2024)
    tr["consensus_spread"] = -3
    va["consensus_spread"] = -3
    with pytest.raises(ValueError, match="Market-derived"):
        FairLineEnsemble().fit(tr, va, ["feature_one", "consensus_spread"])


def test_fair_line_predicts_probability_separate_from_uncertainty():
    tr, va = synthetic_frame(2023, 80), synthetic_frame(2024, 40)
    model = FairLineEnsemble().fit(tr, va, ["feature_one", "feature_two"])
    p = model.predict_frame(va.iloc[:5])
    assert ((p["home_win_probability"] >= 0) & (p["home_win_probability"] <= 1)).all()
    assert (p["prediction_stddev"] > 0).all()
    assert not np.allclose(p["home_win_probability"], p["prediction_stddev"])


def test_rolling_features_are_shifted_before_current_game():
    df = pd.DataFrame({"game_id":["g1","g2","g3"],"season":[2024]*3,"week":[1,2,3],"gameday":pd.to_datetime(["2024-09-01","2024-09-08","2024-09-15"]),"team":["A"]*3,"opponent":["B","C","D"],"starter_qb_id":["q1"]*3,"margin":[100.0,20.0,-10.0]})
    out = add_pregame_rolling_features(df, windows=(2,), metrics=["margin"])
    assert np.isnan(out.loc[0, "pre_margin_r2"])
    assert out.loc[1, "pre_margin_r2"] == 100.0
    assert out.loc[2, "pre_margin_r2"] == 60.0


def test_optional_cpoe_is_not_created_by_team_metrics():
    pbp = pd.DataFrame({"game_id":["g1","g1"],"season":[2024,2024],"week":[1,1],"posteam":["A","B"],"defteam":["B","A"],"play_type":["pass","pass"],"epa":[0.2,-0.1],"success":[1.0,0.0],"yards_gained":[25,3],"down":[1,1],"sack":[0,0],"interception":[0,0]})
    out = team_game_metrics(pbp)
    assert not any("cpoe" in c for c in out.columns)


def test_unplayed_future_game_gets_pregame_row_and_prior_roll():
    schedules = pd.DataFrame([
        {"game_id":"g1","season":2026,"week":1,"gameday":"2026-09-10","home_team":"A","away_team":"B","home_score":30,"away_score":20,"home_rest":7,"away_rest":7,"location":"Home","home_qb_id":"qa","away_qb_id":"qb","home_qb_name":"QA","away_qb_name":"QB"},
        {"game_id":"g2","season":2026,"week":2,"gameday":"2026-09-17","home_team":"A","away_team":"C","home_score":np.nan,"away_score":np.nan,"home_rest":7,"away_rest":7,"location":"Home","home_qb_id":"qa","away_qb_id":"qc","home_qb_name":"QA","away_qb_name":"QC"},
    ])
    games = schedule_team_game_metrics(schedules)
    out = add_pregame_rolling_features(games, windows=(4,), metrics=["margin"])
    a_future = out[(out.game_id=="g2") & (out.team=="A")].iloc[0]
    assert a_future["pre_margin_r4"] == 10.0
    assert np.isnan(a_future["margin"])


def test_clean_features_pass():
    assert_market_blind_features(["home_pass_epa", "away_def_success", "rest_diff", "home_margin_lag4"])


@pytest.mark.parametrize("bad", ["consensus_spread","closing_line","home_moneyline","sportsbook_total","market_implied_prob","away_spread_odds","clv_points"])
def test_market_columns_are_rejected(bad):
    with pytest.raises(ValueError, match="Market-derived"):
        assert_market_blind_features(["home_pass_epa", bad])


def test_matchup_builder_uses_pregame_features_and_no_market_fields():
    teams = pd.DataFrame([{"game_id":"g1","team":"H","opponent":"A","season":2024,"week":2,"pre_pass_epa_per_play_r4":0.2,"starter_change":0.0},{"game_id":"g1","team":"A","opponent":"H","season":2024,"week":2,"pre_pass_epa_per_play_r4":0.1,"starter_change":1.0}])
    sched = pd.DataFrame([{"game_id":"g1","season":2024,"week":2,"gameday":"2024-09-08","home_team":"H","away_team":"A","home_score":24,"away_score":20,"home_rest":7,"away_rest":6,"location":"Home"}])
    out = build_matchup_frame(teams, sched)
    cols = model_feature_columns(out)
    assert "home_margin" not in cols and "total_points" not in cols
    assert all("spread" not in c for c in cols)


def _schedules():
    return pd.DataFrame([
        {"game_id":"2024_01_B_A","season":2024,"game_type":"REG","week":1,"gameday":"2024-09-01","away_team":"B","home_team":"A","away_score":17,"home_score":24,"away_rest":7,"home_rest":7,"spread_line":3.0,"total_line":41.0,"away_qb_id":"qb","home_qb_id":"qa","away_qb_name":"QB","home_qb_name":"QA","location":"Home"},
        {"game_id":"2024_02_C_A","season":2024,"game_type":"REG","week":2,"gameday":"2024-09-08","away_team":"C","home_team":"A","away_score":20,"home_score":27,"away_rest":7,"home_rest":7,"spread_line":2.5,"total_line":44.0,"away_qb_id":"qc","home_qb_id":"qa","away_qb_name":"QC","home_qb_name":"QA","location":"Home"},
    ])


def _pbp():
    rows=[]
    for game_id, week, offense, defense, qb, epa in [("2024_01_B_A",1,"A","B","qa",0.3),("2024_01_B_A",1,"B","A","qb",-0.1),("2024_02_C_A",2,"A","C","qa",0.2),("2024_02_C_A",2,"C","A","qc",0.0)]:
        rows.append({"game_id":game_id,"season":2024,"week":week,"posteam":offense,"defteam":defense,"play_type":"pass","epa":epa,"success":float(epa>0),"yards_gained":25 if epa>0 else 3,"down":1,"sack":0,"interception":0,"passer_player_id":qb,"passer_player_name":qb.upper(),"qb_dropback":1,"qb_epa":epa})
        rows.append({"game_id":game_id,"season":2024,"week":week,"posteam":offense,"defteam":defense,"play_type":"run","epa":epa/2,"success":float(epa>0),"yards_gained":5,"down":2,"sack":0,"interception":0,"passer_player_id":np.nan,"passer_player_name":np.nan,"qb_dropback":0,"qb_epa":np.nan})
    return pd.DataFrame(rows)


def test_fair_feature_pipeline_does_not_emit_market_fields():
    fair = build_feature_frame_from_frames(_schedules(), {2024:_pbp()})
    assert len(fair) == 2 and "spread_line" not in fair.columns
    assert not any("market" in c.lower() or "odds" in c.lower() for c in fair.columns)
    assert fair[fair.week == 2].iloc[0]["home_pre_epa_per_play_r4"] == pytest.approx(0.225)


def test_market_line_is_joined_only_in_historical_backtest_frame():
    frame = build_historical_backtest_frame(_schedules(), {2024:_pbp()})
    assert "spread_line" in frame.columns and list(frame["spread_line"]) == [3.0, 2.5]


def test_missing_cpoe_is_not_fabricated():
    pbp = pd.DataFrame({"game_id":["g1","g1"],"season":[2024,2024],"week":[1,1],"posteam":["A","A"],"play_type":["pass","pass"],"epa":[0.1,0.3],"success":[1.0,1.0],"sack":[0,0],"interception":[0,0],"passer_player_id":["q1","q1"],"passer_player_name":["QB One","QB One"]})
    assert "qb_cpoe" not in qb_game_metrics(pbp).columns


def test_future_starter_uses_prior_qb_games_not_current_game():
    schedules = pd.DataFrame([{"game_id":"g1","season":2026,"week":1,"gameday":"2026-09-10","home_team":"A","away_team":"B","home_qb_id":"qa","away_qb_id":"qb","home_qb_name":"QA","away_qb_name":"QB"},{"game_id":"g2","season":2026,"week":2,"gameday":"2026-09-17","home_team":"A","away_team":"C","home_qb_id":"qa","away_qb_id":"qc","home_qb_name":"QA","away_qb_name":"QC"}])
    qbg = pd.DataFrame([{"game_id":"g1","season":2026,"week":1,"team":"A","qb_id":"qa","qb_name":"QA","qb_epa_per_dropback":0.4,"qb_success_rate":0.6,"qb_sack_rate":0.05,"qb_interception_rate":0.02,"qb_dropbacks":30}])
    out = build_pregame_qb_features(schedules, qbg, windows=(4,))
    w1, w2 = out[(out.game_id=="g1") & (out.team=="A")].iloc[0], out[(out.game_id=="g2") & (out.team=="A")].iloc[0]
    assert w1["pre_qb_starts"] == 0 and w2["pre_qb_starts"] == 1
    assert w2["pre_qb_epa_per_dropback_r4"] == 0.4


def fair(margin=6.0):
    now = datetime.now(timezone.utc)
    return FairPrediction(game_id="g1",model_version="v",generated_at=now,data_as_of=now,home_fair_margin=margin,fair_total=45,projected_home_score=25.5,projected_away_score=19.5,home_win_probability=0.65,prediction_stddev=10,ensemble_disagreement=1,data_quality_score=95,feature_snapshot_hash="abc")


def market(spread):
    return MarketLineSnapshot(game_id="g1", observed_at=datetime.now(timezone.utc), home_spread=spread)


def test_spread_sign_home_favorite_value():
    assert home_cover_edge(6, -4) == 2
    ev = evaluate_edge(fair(6), market(-4), qb_availability_confidence=1.0)
    assert ev.recommended_side == "HOME" and ev.edge_points == 2


def test_spread_sign_market_home_favorite_too_expensive():
    assert home_cover_edge(1, -3) == -2
    ev = evaluate_edge(fair(1), market(-3), qb_availability_confidence=1.0)
    assert ev.recommended_side == "AWAY" and ev.edge_points == 2


def test_ats_grade_favorite_and_underdog():
    assert grade_side(7, -4, "HOME") == "WIN"
    assert grade_side(3, -4, "AWAY") == "WIN"
    assert grade_side(4, -4, "HOME") == "PUSH"


def test_unknown_qb_availability_cannot_qualify():
    ev = evaluate_edge(fair(10), market(-4), qb_availability_confidence=None)
    assert ev.qualification_status in {"LEAN", "NO_PLAY"}
    assert any("QB availability" in c for c in ev.top_concerns)


def test_threshold_selection_rejects_2025_holdout():
    df = pd.DataFrame({"season":[2025]*30,"edge_abs":[2.0]*30,"ats_result":["WIN"]*30})
    with pytest.raises(ValueError, match="validation seasons only"):
        select_edge_thresholds_from_validation(df, DEFAULT_CONFIG)


def test_nflverse_spread_sign_is_normalized_to_sportsbook_notation():
    assert nflverse_spread_to_market_home_spread(4.5) == -4.5
    assert nflverse_spread_to_market_home_spread(-3.0) == 3.0


def test_default_holdout_is_2025_and_disjoint():
    assert DEFAULT_CONFIG.holdout_seasons == (2025,)
    assert not (set(DEFAULT_CONFIG.holdout_seasons) & set(DEFAULT_CONFIG.training_seasons))
    assert not (set(DEFAULT_CONFIG.holdout_seasons) & set(DEFAULT_CONFIG.validation_seasons))


def test_threshold_selection_never_uses_holdout():
    selected = set(threshold_selection_seasons(DEFAULT_CONFIG))
    assert selected == {2024} and not selected & set(DEFAULT_CONFIG.holdout_seasons)


def test_overlapping_config_raises():
    with pytest.raises(ValueError):
        ModelConfig(training_seasons=(2023,2024), validation_seasons=(2024,), holdout_seasons=(2025,))
