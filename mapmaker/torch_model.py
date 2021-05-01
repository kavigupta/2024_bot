import attr

import torch
import torch.nn as nn

import numpy as np

from .model import Model
from .trend_model import StableTrendModel


class DemographicCategoryPredictor(nn.Module):
    def __init__(self, f, d, years, gamma=0.5):
        super().__init__()
        self.f = f
        self.d = d
        self.gamma = gamma
        self.years = years
        self.latent_demographic_model = nn.Sequential(nn.Linear(f, d), nn.Softmax(-1))
        self.turnout_heads = nn.Parameter(torch.randn(len(years), d, 1))
        self.partisanship_heads = nn.Parameter(torch.randn(len(years), d, 1))

    def get_heads(self, idxs):
        turnout, partisanship = (
            torch.sigmoid(self.turnout_heads)[idxs],
            torch.tanh(self.partisanship_heads)[idxs],
        )
        return turnout, turnout * partisanship

    def forward(self, years, features):
        assert len(features) == len(years)
        idxs = [self.years.index(y) for y in years]
        features = torch.tensor(features).float()
        # import IPython; IPython.embed()
        demos = self.latent_demographic_model(features)
        turnout_heads, partisanship_heads = self.get_heads(idxs)
        return (
            torch.bmm(demos, turnout_heads).squeeze(-1),
            torch.bmm(demos, partisanship_heads).squeeze(-1),
        )

    def loss(self, years, features, target_turnouts, target_partisanships, cvaps):

        target_turnouts = np.array(target_turnouts)
        target_partisanships = np.array(target_partisanships)
        target_tp = target_turnouts * target_partisanships
        target_t = torch.tensor(target_turnouts).float()
        target_tp = torch.tensor(target_tp).float()
        cvaps = torch.tensor(np.array(cvaps)).float()
        t, tp = self(years, features)
        loss = (target_t - t) ** 2 * self.gamma + (target_tp - tp) ** 2
        return (loss * cvaps).sum() / cvaps.sum()

    def predict(self, year, features):
        t, tp = self([year], [features])
        return (tp / t).detach().numpy()[0], t.detach().numpy()[0]

    @staticmethod
    def train(
        years,
        features,
        target_turnouts,
        target_partisanships,
        cvaps,
        iters=1000,
        lr=1e-2,
        *,
        dimensions,
    ):
        torch.manual_seed(0)
        dcm = DemographicCategoryPredictor(dimensions + 1, 10, years)
        opt = torch.optim.Adam(dcm.parameters(), lr=lr)
        for itr in range(iters):
            opt.zero_grad()
            lv = dcm.loss(years, features, target_turnouts, target_partisanships, cvaps)
            if (itr + 1) % 100 == 0:
                print(itr, lv.item())
            lv.backward()
            opt.step()
        return dcm


@attr.s
class AdjustedDemographicCategoryModel:
    dcm = attr.ib()
    residuals = attr.ib()
    trend_model = attr.ib()

    @staticmethod
    def train(*, years, features, data, feature_kwargs):
        turnouts = {y: data[y].total_votes / data[y].CVAP for y in years}
        dcm = DemographicCategoryPredictor.train(
            years,
            [features.features(y) for y in years],
            [turnouts[y] for y in years],
            [data[y].dem_margin for y in years],
            [data[y].CVAP for y in years],
            iters=6000,
            **feature_kwargs,
        )
        residuals = {}
        for y in years:
            p, t = dcm.predict(y, features.features(y))
            residuals[y] = data[y].dem_margin - p, turnouts[y] - t
        return AdjustedDemographicCategoryModel(dcm, residuals, StableTrendModel(0))

    def predict(self, *, model_year, year, features, correct):
        p, t = self.dcm.predict(model_year, features)
        if correct:
            pr = self.trend_model(
                features,
                self.residuals[model_year][0],
                year=year,
                base_year=model_year,
            )
            if correct == "just_residuals":
                p = pr
            else:
                p = p + pr

            t = t + self.residuals[model_year][1]
        return p, t


class DemographicCategoryModel(Model):
    def __init__(self, data_by_year, feature_kwargs={}):
        super().__init__(data_by_year, feature_kwargs)
        self.adcm = AdjustedDemographicCategoryModel.train(
            years=sorted(y for y in data_by_year if y <= 2020),
            features=self.features,
            data=self.data,
            feature_kwargs=feature_kwargs,
        )

    def fully_random_sample(self, *, year, prediction_seed, correct):
        # use the 2020 predictor since that's the best we have
        # TODO ADD THE PERTURBATIONS
        model_year = 2020 if year == 2024 else year
        return self.adcm.predict(
            model_year=model_year,
            year=year,
            features=self.features.features(year),
            correct=correct,
        )
