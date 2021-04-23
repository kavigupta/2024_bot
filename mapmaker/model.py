from abc import ABC, abstractmethod

import numpy as np
import attr
import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .stitch_map import generate_map
from .aggregation import get_electoral_vote
from .features import Features, metadata


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

    def with_bias(self, x):
        return LinearModel(self.weights, self.residuals, x, self.trend_model)

    @staticmethod
    def train(
        features, margin, total_votes, bias=0, trend_model=StableTrendModel(0)
    ):
        weights = (
            LinearRegression(fit_intercept=False)
            .fit(features, margin, sample_weight=total_votes)
            .coef_
        )
        residuals = margin - features @ weights
        return LinearModel(weights, residuals, bias, trend_model)

    def predict(self, features, correct=True, *, year):
        pred = features @ self.weights
        if correct:
            pred = (
                pred + self.trend_model(features, self.residuals, year=year) + self.bias
            )
        return np.clip(pred, -0.9, 0.9)

    def perturb(self, seed, alpha):
        rng = np.random.RandomState(seed)
        noise = rng.randn(*self.weights.shape)
        noise = noise * alpha * np.abs(self.weights).mean()
        trend_model = NoisedTrendModel.of(rng, len(self.weights))
        return LinearModel(self.weights + noise, self.residuals, self.bias, trend_model)


def compute_ec_bias(predictor, data, features, alpha):
    overall = []
    for seed in range(1000):
        predictions = predictor.perturb(seed, alpha).predict(
            features, correct=True, year=2024
        )
        dem, gop = get_electoral_vote(data, dem_margin=predictions)
        if dem == gop:
            continue
        overall += [dem > gop]
    return np.mean(overall)


class Model:
    def __init__(self, data_by_year, feature_kwargs={}, *, alpha):
        self.metadata = metadata(data_by_year, train_key=2020)
        self.features = Features.fit(data_by_year, train_key=2020, **feature_kwargs)
        self.predictor = LinearModel.train(
            self.features.features(2020),
            self.metadata.biden_2020,
            self.metadata.total_votes,
        )
        self.alpha = alpha

    def unbias_predictor(self, *, for_year):
        starting_bias = compute_ec_bias(
            self.predictor.with_bias(0),
            self.metadata,
            self.features.features(for_year),
            self.alpha,
        )
        print(
            f"Without correction, democrats win the EC {starting_bias:.2%} of the time"
        )

        bias_values = np.arange(-0.02, -0.005, 0.001)
        biases = np.array(
            [
                compute_ec_bias(
                    self.predictor.with_bias(x),
                    self.metadata,
                    self.features.features(for_year),
                    self.alpha,
                )
                for x in tqdm.tqdm(bias_values)
            ]
        )
        idx = np.argmin(np.abs(biases - 0.5))
        best_bias = bias_values[idx]
        print(
            f"Computed best bias: {best_bias:.2%}, which gives democrats an EC win {biases[idx]:.0%} of the time"
        )
        self.predictor = self.predictor.with_bias(best_bias)

    def win_consistent_with(self, predictions, seed):
        if seed is None:
            return True
        dem, gop = get_electoral_vote(self.metadata, dem_margin=predictions)
        dem_win = dem > gop  # ties go to gop
        # even days, democrat. odd days, gop
        return dem_win == (seed % 2 == 0)

    def sample(self, *, year, seed=None, correct=True):
        rng = np.random.RandomState(seed)
        while True:
            predictor = self.predictor
            if seed is not None:
                predictor = predictor.perturb(rng.randint(2 ** 32), self.alpha)
            predictions = predictor.predict(
                self.features.features(year), correct, year=year
            )
            if self.win_consistent_with(predictions, seed):
                break
        return predictions

    def sample_map(self, title, path, **kwargs):
        print(f"Generating {title}")
        predictions = self.sample(**kwargs)
        return generate_map(self.metadata, title, path, dem_margin=predictions)
