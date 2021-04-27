from abc import ABC, abstractmethod

import numpy as np
import attr

import copy

from sklearn.linear_model import LinearRegression

from .model import Model


class TrendModel(ABC):
    @abstractmethod
    def extra_residue(self, features, residuals):
        pass

    def __call__(self, features, residuals, *, year):
        assert year in {2020, 2024}
        return residuals + self.extra_residue(features, residuals) * ((year - 2020) / 4)


@attr.s
class StableTrendModel(TrendModel):
    trendiness = attr.ib()

    def extra_residue(self, features, residuals):
        return residuals * self.trendiness


@attr.s
class NoisedTrendModel(TrendModel):
    trendiness_by_feature = attr.ib()
    trend_mu = attr.ib()
    trend_sigma = attr.ib()

    @staticmethod
    def of(rng, n_features, trend_mu_mean=0, trend_mu_sigma=0.2, trend_sigma=0.1):
        return NoisedTrendModel(
            rng.randn(n_features),
            rng.randn() * trend_mu_sigma + trend_mu_mean,
            trend_sigma,
        )

    def extra_residue(self, features, residuals):
        trendiness = features @ self.trendiness_by_feature
        trendiness = (trendiness - trendiness.mean()) / trendiness.std()
        trendiness = trendiness * self.trend_sigma + self.trend_mu
        return residuals * trendiness


@attr.s
class LinearModel:
    weights = attr.ib()
    residuals = attr.ib()
    bias = attr.ib()
    trend_model = attr.ib()
    clip_range = attr.ib()

    def with_bias(self, x):
        return LinearModel(
            self.weights, self.residuals, x, self.trend_model, self.clip_range
        )

    @staticmethod
    def train(
        features,
        margin,
        total_votes,
        bias=0,
        trend_model=StableTrendModel(0),
        *,
        clip_range,
    ):
        weights = (
            LinearRegression(fit_intercept=False)
            .fit(features, margin, sample_weight=total_votes)
            .coef_
        )
        residuals = margin - features @ weights
        return LinearModel(weights, residuals, bias, trend_model, clip_range)

    def predict(self, features, correct=True, *, year):
        pred = features @ self.weights
        if correct:
            pred = (
                pred + self.trend_model(features, self.residuals, year=year) + self.bias
            )
        return np.clip(pred, *self.clip_range)

    def perturb(self, seed, alpha, *, noise_trends):
        rng = np.random.RandomState(seed % (2 ** 32))
        noise = rng.randn(*self.weights.shape)
        noise = noise * alpha * np.abs(self.weights)
        if noise_trends:
            trend_model = NoisedTrendModel.of(rng, len(self.weights))
        else:
            trend_model = self.trend_model
        return LinearModel(
            self.weights + noise,
            self.residuals,
            self.bias,
            trend_model,
            self.clip_range,
        )


class LinearMixtureModel(Model):
    def __init__(self, data_by_year, feature_kwargs={}):
        super().__init__(data_by_year, feature_kwargs)
        self.predictor = LinearModel.train(
            self.features.features(2020),
            self.metadata.biden_2020,
            self.metadata.CVAP,
            clip_range=(-0.9, 0.9),
        )
        self.turnout_predictor = LinearModel.train(
            self.features.features(2020),
            self.metadata.turnout,
            self.metadata.CVAP,
            clip_range=(0.2, 0.9),
        )

    def with_predictor(self, predictor):
        self = copy.copy(self)
        self.predictor = predictor
        return self

    def fully_random_sample(self, *, year, prediction_seed, correct):
        predictor = self.predictor
        turnout_predictor = self.turnout_predictor
        if prediction_seed is not None:
            predictor = predictor.perturb(
                2 * prediction_seed, self.alpha, noise_trends=True
            )
            turnout_predictor = turnout_predictor.perturb(
                2 * prediction_seed + 1, 1 / 3 * self.alpha, noise_trends=False
            )
        features = self.features.features(year)
        predictions = predictor.predict(features, correct, year=year)
        turnout = turnout_predictor.predict(features, correct, year=year)
        return predictions, turnout
