import numpy as np
import pandas as pd

from beard_model.qv2 import Q_FEATURES, qualifier_pipeline, _fixed_gate_summary
from beard_model.integrity import market_derived_columns


def test_qv2_is_explicitly_post_market_not_fair_line_contract():
    assert "market_abs_spread" in Q_FEATURES
    assert "market_abs_spread" in market_derived_columns(Q_FEATURES)


def test_qualifier_pipeline_outputs_probabilities():
    x = pd.DataFrame({c: [0.0, 1.0, 2.0, 3.0] for c in Q_FEATURES})
    y = np.array([0, 0, 1, 1])
    m = qualifier_pipeline(0.1).fit(x, y)
    p = m.predict_proba(x)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_fixed_probability_gate_does_not_optimize_on_results():
    test = pd.DataFrame({"ats_result": ["WIN", "LOSS", "WIN", "LOSS"]})
    p = np.array([0.60, 0.59, 0.54, 0.40])
    s = _fixed_gate_summary(test, p, 0.55)
    assert s["bets"] == 2
    assert s["wins"] == 1
    assert s["losses"] == 1
