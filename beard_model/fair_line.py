from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .integrity import assert_market_blind_features


@dataclass
class TrainedTarget:
    ridge: Pipeline
    booster: Pipeline
    ridge_alpha: float
    booster_params: dict
    ridge_weight: float
    booster_weight: float
    validation_residual_std: float
    validation_mae: float


class FairLineEnsemble:
    """Market-blind ensemble for home margin and total points."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_columns: list[str] = []
        self.margin_model: TrainedTarget | None = None
        self.total_model: TrainedTarget | None = None
        self.model_version = "beard-v0.1-dev"

    @staticmethod
    def _ridge(alpha: float) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    def _booster(self, params: dict) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("model", HistGradientBoostingRegressor(random_state=self.random_seed, **params)),
        ])

    def _fit_target(self, train: pd.DataFrame, val: pd.DataFrame, target: str) -> TrainedTarget:
        xtr, ytr = train[self.feature_columns], train[target].astype(float)
        xva, yva = val[self.feature_columns], val[target].astype(float)
        ridge_candidates = [1.0, 10.0, 50.0, 100.0]
        best_ridge = None
        for alpha in ridge_candidates:
            m = self._ridge(alpha)
            m.fit(xtr, ytr)
            pred = m.predict(xva)
            score = mean_absolute_error(yva, pred)
            if best_ridge is None or score < best_ridge[0]:
                best_ridge = (score, alpha, m, pred)

        booster_candidates = [
            {"learning_rate": 0.04, "max_iter": 200, "max_leaf_nodes": 15, "l2_regularization": 2.0},
            {"learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 15, "l2_regularization": 5.0},
            {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 5.0},
        ]
        best_booster = None
        for params in booster_candidates:
            m = self._booster(params)
            m.fit(xtr, ytr)
            pred = m.predict(xva)
            score = mean_absolute_error(yva, pred)
            if best_booster is None or score < best_booster[0]:
                best_booster = (score, params, m, pred)

        r_mae, alpha, ridge, rpred = best_ridge
        b_mae, bparams, booster, bpred = best_booster
        inv_r, inv_b = 1.0 / max(r_mae, 1e-9), 1.0 / max(b_mae, 1e-9)
        rw = inv_r / (inv_r + inv_b)
        bw = 1.0 - rw
        blend = rw * rpred + bw * bpred
        residual = yva.to_numpy() - blend
        resid_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 14.0
        return TrainedTarget(
            ridge=ridge, booster=booster, ridge_alpha=alpha, booster_params=bparams,
            ridge_weight=float(rw), booster_weight=float(bw), validation_residual_std=max(resid_std, 1.0),
            validation_mae=float(mean_absolute_error(yva, blend)),
        )

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame, feature_columns: list[str]) -> "FairLineEnsemble":
        assert_market_blind_features(feature_columns)
        forbidden_targets = {"home_margin", "total_points"}
        if forbidden_targets & set(feature_columns):
            raise ValueError("Outcome targets cannot be included in the fair-line feature matrix.")
        missing = sorted((set(feature_columns) | forbidden_targets) - set(train.columns))
        if missing:
            raise ValueError(f"Training frame missing columns: {', '.join(missing)}")
        missing_val = sorted((set(feature_columns) | forbidden_targets) - set(validation.columns))
        if missing_val:
            raise ValueError(f"Validation frame missing columns: {', '.join(missing_val)}")
        self.feature_columns = list(feature_columns)
        self.margin_model = self._fit_target(train, validation, "home_margin")
        self.total_model = self._fit_target(train, validation, "total_points")
        return self

    @staticmethod
    def _blend(target: TrainedTarget, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rp = target.ridge.predict(x)
        bp = target.booster.predict(x)
        blend = target.ridge_weight * rp + target.booster_weight * bp
        disagreement = np.abs(rp - bp)
        return blend, disagreement, np.column_stack([rp, bp])

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.margin_model is None or self.total_model is None:
            raise RuntimeError("FairLineEnsemble must be fit before prediction.")
        assert_market_blind_features(self.feature_columns)
        x = frame[self.feature_columns]
        margin, disagreement, _ = self._blend(self.margin_model, x)
        total, total_disagreement, _ = self._blend(self.total_model, x)
        observed = x.notna().sum(axis=1).to_numpy()
        quality = 100.0 * observed / max(len(self.feature_columns), 1)
        base_std = self.margin_model.validation_residual_std
        pred_std = np.sqrt(base_std ** 2 + disagreement ** 2)
        win_prob = norm.cdf(margin / np.maximum(pred_std, 1e-6))
        home_score = (total + margin) / 2.0
        away_score = (total - margin) / 2.0
        out = pd.DataFrame(index=frame.index)
        for c in ["game_id", "season", "week", "gameday", "home_team", "away_team"]:
            if c in frame.columns:
                out[c] = frame[c]
        out["home_fair_margin"] = margin
        out["fair_total"] = total
        out["projected_home_score"] = home_score
        out["projected_away_score"] = away_score
        out["home_win_probability"] = win_prob
        out["prediction_stddev"] = pred_std
        out["ensemble_disagreement"] = disagreement
        out["total_ensemble_disagreement"] = total_disagreement
        out["data_quality_score"] = quality
        return out

    def refit_on_train_plus_validation(self, train: pd.DataFrame, validation: pd.DataFrame) -> "FairLineEnsemble":
        """Refit selected hyperparameters on train+validation after validation decisions are frozen."""
        if self.margin_model is None or self.total_model is None:
            raise RuntimeError("Fit on train/validation before refitting for holdout.")
        combined = pd.concat([train, validation], ignore_index=True)
        x = combined[self.feature_columns]
        for target_name, trained in (("home_margin", self.margin_model), ("total_points", self.total_model)):
            y = combined[target_name].astype(float)
            ridge = self._ridge(trained.ridge_alpha)
            booster = self._booster(trained.booster_params)
            ridge.fit(x, y)
            booster.fit(x, y)
            trained.ridge = ridge
            trained.booster = booster
        return self

    def metadata_hash(self) -> str:
        payload = {
            "features": self.feature_columns,
            "margin_alpha": None if self.margin_model is None else self.margin_model.ridge_alpha,
            "margin_booster": None if self.margin_model is None else self.margin_model.booster_params,
            "total_alpha": None if self.total_model is None else self.total_model.ridge_alpha,
            "total_booster": None if self.total_model is None else self.total_model.booster_params,
            "seed": self.random_seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
