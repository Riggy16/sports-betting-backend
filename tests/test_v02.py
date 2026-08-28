import numpy as np
import pandas as pd

from beard_model.v02 import add_dynamic_margin_rating, add_v02_features, V02_FEATURES, V02Spec
from beard_model.integrity import assert_market_blind_features


def test_dynamic_rating_is_pregame_and_market_blind():
    frame=pd.DataFrame([
        {"season":2024,"gameday":pd.Timestamp("2024-09-01"),"game_id":"g1","home_team":"A","away_team":"B","home_margin":10.0,"home_field":1.0},
        {"season":2024,"gameday":pd.Timestamp("2024-09-08"),"game_id":"g2","home_team":"A","away_team":"C","home_margin":-3.0,"home_field":1.0},
    ])
    out=add_dynamic_margin_rating(frame,V02Spec(rating_k=.2,offseason_regression=.5,home_advantage_points=1.5))
    assert out.loc[0,"v02_dynamic_margin_baseline"] == 1.5
    assert out.loc[1,"v02_dynamic_rating_diff"] != 0


def test_v02_feature_contract_has_no_market_fields():
    assert len(V02_FEATURES) <= 30
    assert_market_blind_features(V02_FEATURES)


def test_correct_pass_matchup_math_rewards_weak_opponent_defense():
    from beard_model.v02 import CORE_SOURCE_COLUMNS
    d={c:0.0 for c in CORE_SOURCE_COLUMNS}
    d.update({"season":2024,"gameday":pd.Timestamp("2024-09-01"),"game_id":"g","home_team":"H","away_team":"A","home_margin":0.0,"spread_line":0.0})
    d["home_field"]=1.0
    d["home_pre_pass_epa_per_play_r8"]=0.2
    d["away_pre_pass_epa_per_play_allowed_r8"]=0.1
    d["away_pre_pass_epa_per_play_r8"]=0.0
    d["home_pre_pass_epa_per_play_allowed_r8"]=-0.1
    out=add_v02_features(pd.DataFrame([d]))
    assert out.loc[0,"v02_matchup_pass_epa"] == 0.4
