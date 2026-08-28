import numpy as np
import pandas as pd

from beard_model.grading import MODEL_FEATURES, add_grade_features, apply_grade_reference
from beard_model.integrity import assert_market_blind_features


def _frame():
    rows = []
    for i in range(6):
        row = {
            'season': 2020 + i // 3, 'gameday': f'202{ i // 3 }-09-{10+i:02d}', 'game_id': f'g{i}',
            'home_team': 'H', 'away_team': 'A', 'home_margin': float(i-2), 'home_field': 1.0,
            'rest_diff': float(i % 3 - 1), 'spread_line': 2.5,
            'home_starter_change': 0.0, 'away_starter_change': 1.0 if i == 2 else 0.0,
        }
        names = {
            'epa_per_play_r8': .10, 'epa_per_play_allowed_r8': -.05,
            'success_rate_r8': .46, 'success_rate_allowed_r8': .43,
            'pass_epa_per_play_r8': .14, 'pass_epa_per_play_allowed_r8': -.03,
            'rush_epa_per_play_r8': .02, 'rush_epa_per_play_allowed_r8': -.01,
            'explosive_pass_rate_r8': .12, 'explosive_pass_rate_allowed_r8': .09,
            'sack_rate_r8': .06, 'sack_rate_generated_r8': .08,
            'interception_rate_r8': .025, 'interception_rate_generated_r8': .03,
            'margin_r4': 4.0, 'margin_r8': 2.0,
            'qb_epa_per_dropback_r8': .16, 'qb_success_rate_r8': .48,
            'qb_cpoe_r8': 2.0, 'qb_sack_rate_r8': .05, 'qb_starts': 30.0,
        }
        for side, sign in [('home', 1), ('away', -1)]:
            for name, value in names.items():
                row[f'{side}_pre_{name}'] = value + sign * 0.01 * i
        rows.append(row)
    return pd.DataFrame(rows)


def test_grading_model_features_are_market_blind():
    assert_market_blind_features(MODEL_FEATURES)


def test_grades_are_deterministic_for_same_reference():
    raw = add_grade_features(_frame())
    a = apply_grade_reference(raw.iloc[[4]].copy(), raw.iloc[:4].copy())
    b = apply_grade_reference(raw.iloc[[4]].copy(), raw.iloc[:4].copy())
    cols = [c for c in a.columns if c.startswith('grade_')]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_better_home_pass_offense_improves_pass_matchup_grade():
    frame = _frame()
    raw = add_grade_features(frame)
    ref = raw.iloc[:4].copy()
    base = apply_grade_reference(raw.iloc[[4]].copy(), ref)
    boosted_raw = raw.iloc[[4]].copy()
    boosted_raw['raw_home_pass_offense'] += 1.0
    boosted = apply_grade_reference(boosted_raw, ref)
    assert boosted.iloc[0]['grade_pass_matchup'] > base.iloc[0]['grade_pass_matchup']


def test_no_spread_needed_to_build_grades():
    frame = _frame().drop(columns=['spread_line'])
    raw = add_grade_features(frame)
    graded = apply_grade_reference(raw.iloc[[4]].copy(), raw.iloc[:4].copy())
    assert np.isfinite(graded.iloc[0]['grade_home_team'])
    assert np.isfinite(graded.iloc[0]['grade_pass_matchup'])
