import numpy as np
import attr
import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .stitch_map import generate_map
from .processing import get_electoral_vote


@attr.s
class LinearModel:
    weights = attr.ib()
    residuals = attr.ib()
    bias = attr.ib()

    def with_bias(self, x):
        return LinearModel(self.weights, self.residuals, x)

    @staticmethod
    def train(features, margin, total_votes, bias=0):
        weights = (
            LinearRegression(fit_intercept=False)
            .fit(features, margin, sample_weight=total_votes)
            .coef_
        )
        residuals = margin - features @ weights
        return LinearModel(weights, residuals, bias)

    def predict(self, features, correct=True):
        pred = features @ self.weights
        if correct:
            pred = pred + self.residuals + self.bias
        return np.clip(pred, -0.8, 0.8)

    def perturb(self, seed, alpha):
        noise = np.random.RandomState(seed).randn(*self.weights.shape)
        noise = noise * alpha * np.abs(self.weights)
        return LinearModel(self.weights + noise, self.residuals, self.bias)


def compute_ec_bias(predictor, data, features, alpha):
    data = data.copy()
    overall = []
    for seed in range(100):
        data["temp"] = predictor.perturb(seed, alpha).predict(features, correct=True)
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
            self.features, data.biden_2020, data.total_votes
        )
        self.alpha = alpha

    def unbias_predictor(self):
        bias_values = np.linspace(-0.05, 0.05, 11)
        biases = np.array(
            [
                compute_ec_bias(
                    self.predictor.with_bias(x), self.data, self.features, self.alpha
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

    def sample(self, title, path, seed=None, correct=True):
        predictor = self.predictor
        if seed is not None:
            predictor = predictor.perturb(seed, self.alpha)
        data = self.data.copy()
        data["temp"] = predictor.predict(self.features, correct)
        generate_map(data, "temp", title, path)


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def get_features(data, pca=20):
    features = data.fillna(0).copy()
    features = features[
        [x for x in features if x not in {"FIPS", "biden_2020", "total_votes", "state"}]
    ]
    features = np.array(features)
    if pca is not None:
        features = PCA(20, whiten=False).fit_transform(features)
    return add_ones(features)
