from abc import ABC, abstractmethod

import numpy as np
import attr
import tqdm

from sklearn.linear_model import LinearRegression

from .data import ec
from .stitch_map import generate_map
from .aggregation import get_electoral_vote, get_state_results
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
    def train(features, margin, total_votes, bias=0, trend_model=StableTrendModel(0)):
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
        noise = noise * alpha * np.abs(self.weights)
        trend_model = NoisedTrendModel.of(rng, len(self.weights))
        return LinearModel(self.weights + noise, self.residuals, self.bias, trend_model)


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
        state_preds = self.family_of_predictions(year=for_year)

        def bias(preds):
            dems = preds > 0
            gop = preds < 0
            ecs = np.array(ec())[:, 0]
            return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()

        print(
            f"Without correction, democrats win the EC {bias(state_preds):.2%} of the time"
        )

        bias_values = np.arange(-0.02, -0.005, 0.001)
        biases = np.array([bias(state_preds + x) for x in tqdm.tqdm(bias_values)])

        idx = np.argmin(np.abs(biases - 0.5))
        best_bias = bias_values[idx]
        print(
            f"Computed best bias: {best_bias:.2%}, which gives democrats an EC win {biases[idx]:.0%} of the time"
        )
        self.predictor = self.predictor.with_bias(best_bias)

    def family_of_predictions(self, *, year, correct=True, n_seeds=1000):
        state_results = []
        for seed in range(n_seeds):
            predictions = self.fully_random_sample(
                year=year, correct=correct, prediction_seed=seed
            )
            state_results.append(
                get_state_results(self.metadata, dem_margin=predictions)
            )
        return np.array(state_results)

    def fully_random_sample(self, *, year, prediction_seed, correct):
        predictor = self.predictor
        if prediction_seed is not None:
            predictor = predictor.perturb(prediction_seed, self.alpha)
        return predictor.predict(self.features.features(year), correct, year=year)

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
            predictions = self.fully_random_sample(
                year=year,
                prediction_seed=rng.randint(2 ** 32) if seed is not None else None,
                correct=correct,
            )
            if self.win_consistent_with(predictions, seed):
                break
        return predictions

    def sample_map(self, title, path, **kwargs):
        print(f"Generating {title}")
        predictions = self.sample(**kwargs)
        return generate_map(self.metadata, title, path, dem_margin=predictions)
