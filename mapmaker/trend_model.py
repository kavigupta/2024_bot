from abc import ABC, abstractmethod

import attr


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
