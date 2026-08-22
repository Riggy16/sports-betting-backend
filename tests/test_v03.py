import numpy as np
import pandas as pd

from beard_model.v03 import residual_target, qualify_predictions, V03_FEATURES


def test_residual_target_matches_nflverse_fair_margin_convention():
    f = pd.DataFrame({"home_margin": [7.0, -1.0], "spread_line": [4.5, -3.0]})
    r = residual_target(f)
    assert list(r) == [2.5, 2.0]


def test_qualification_requires_probability_and_residual_direction_agreement():
    out = qualify_predictions(
        np.array([0.60, 0.40, 0.60, 0.51]),
        np.array([2.0, -2.0, -2.0, 3.0]),
    )
    assert out.loc[0, "status"] in {"QUALIFIED", "STRONG"}
    assert out.loc[0, "side"] == "HOME"
    assert out.loc[1, "side"] == "AWAY"
    assert out.loc[2, "status"] == "NO_PLAY"
    assert out.loc[2, "side"] == "NONE"
    assert out.loc[3, "status"] == "NO_PLAY"


def test_v03_is_explicitly_post_market():
    assert "market_fair_margin" in V03_FEATURES
    assert "market_abs_spread" in V03_FEATURES
