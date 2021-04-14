import numpy as np
import attr
import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .stitch_map import generate_map
from .processing import get_electoral_vote


@attr.s
class StableTrendModel:
    trendiness = attr.ib()

    def __call__(self, features, residuals):
        return residuals * (1 + self.trendiness)


@attr.s
class NoisedTrendModel:
    trendiness_by_feature = attr.ib()
    trend_mu = attr.ib()
    trend_sigma = attr.ib()

    @staticmethod
    def of(rng, n_features, trend_mu_mean=0.25, trend_mu_sigma=0.2, trend_sigma=0.1):
        return NoisedTrendModel(
            rng.randn(n_features),
            rng.randn() * trend_mu_sigma + trend_mu_mean,
            trend_sigma,
        )

    def __call__(self, features, residuals):
        trendiness = features @ self.trendiness_by_feature
        trendiness = (trendiness - trendiness.mean()) / trendiness.std()
        trendiness = trendiness * self.trend_sigma + self.trend_mu
        return residuals * (1 + trendiness)


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
        features, margin, total_votes, bias=0, trend_model=StableTrendModel(0.25)
    ):
        weights = (
            LinearRegression(fit_intercept=False)
            .fit(features, margin, sample_weight=total_votes)
            .coef_
        )
        residuals = margin - features @ weights
        return LinearModel(weights, residuals, bias, trend_model)

    def predict(self, features, correct=True, adjust=True):
        pred = features @ self.weights
        if correct:
            pred = pred + self.residuals + self.bias
        elif adjust:
            pred = pred + self.trend_model(features, self.residuals) + self.bias
        return np.clip(pred, -0.8, 0.8)

    def perturb(self, seed, alpha):
        rng = np.random.RandomState(seed)
        noise = rng.randn(*self.weights.shape)
        noise = noise * alpha * np.abs(self.weights)
        trend_model = NoisedTrendModel.of(rng, len(self.weights))
        return LinearModel(self.weights + noise, self.residuals, self.bias, trend_model)


def compute_ec_bias(predictor, data, features, alpha):
    data = data.copy()
    overall = []
    for seed in range(100):
        data["temp"] = predictor.perturb(seed, alpha).predict(
            features, correct=True, adjust=True
        )
        dem, gop = get_electoral_vote(data, "temp")
        if dem == gop:
            continue
        overall += [dem > gop]
    return np.mean(overall)


class Model:
    def __init__(self, data, feature_kwargs={}, *, alpha):
        self.data = data
        self.features = get_features(data, **feature_kwargs)
        self.predictor = LinearModel.train(
            self.run_pca(self.data), data.biden_2020, data.total_votes
        )
        self.alpha = alpha

    def run_pca(self, data):
        return add_ones(self.features.transform(strip_columns(data)))

    def unbias_predictor(self):
        bias_values = np.array([0])
        biases = np.array(
            [
                compute_ec_bias(
                    self.predictor.with_bias(x),
                    self.data,
                    self.run_pca(self.data),
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

    def sample(self, title, path, data, seed=None, correct=True, adjust=True):
        predictor = self.predictor
        if seed is not None:
            predictor = predictor.perturb(seed, self.alpha)
        data = data.copy()
        data["temp"] = predictor.predict(self.run_pca(data), correct, adjust)
        return generate_map(data, "temp", title, path)


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def strip_columns(data):
    features = data.fillna(0).copy()
    features = features[
        [x for x in features if x not in {"FIPS", "biden_2020", "total_votes", "state"}]
    ]
    return np.array(features)


def get_features(data, pca=20):
    features = strip_columns(data)
    if pca is not None:
        features = PCA(pca, whiten=True).fit(features)
    return features
