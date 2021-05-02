import attr

import torch
import torch.nn as nn

import numpy as np

from permacache import permacache, stable_hash

from .model import Model
from .trend_model import StableTrendModel
from .utils import hash_model


class DemographicCategoryPredictor(nn.Module):
    # to refresh cache, increment this
    version = 1.1

    def __init__(self, f, d, years, previous_partisanships, gamma=0.5):
        super().__init__()
        self.f = f
        self.d = d
        self.gamma = gamma
        self.years = years
        self.min_turn = 0.4
        self.max_turn = 0.75
        assert set(years) - set(previous_partisanships) == set()
        self.previous_partisanships = previous_partisanships
        self.latent_demographic_model = nn.Sequential(nn.Linear(f, d), nn.Softmax(-1))
        self.turnout_heads = nn.Parameter(torch.randn(len(years), d, 1))
        self.partisanship_heads = nn.Parameter(torch.randn(len(years), d, 1))

    def get_heads(
        self,
        idxs,
        partisanship_noise=0,
        turnout_noise=0,
        turnout_weights=None,
    ):
        turnout, partisanship = (
            torch.sigmoid(self.turnout_heads + turnout_noise)
            * (self.max_turn - self.min_turn)
            + self.min_turn,
            torch.tanh(self.partisanship_heads + partisanship_noise),
        )
        partisanship = partisanship[idxs]
        if turnout_weights is not None:
            assert len(idxs) == 1
            turnout = (turnout * turnout_weights[:, None, None]).sum(0)[None]
        else:
            turnout = turnout[idxs]
        return turnout, turnout * partisanship

    def forward(self, years, features, **kwargs):
        assert len(features) == len(years)
        idxs = [self.years.index(y) for y in years]
        features = torch.tensor(features).float()
        # import IPython; IPython.embed()
        demos = self.latent_demographic_model(features)
        turnout_heads, partisanship_heads = self.get_heads(idxs, **kwargs)
        t, tp = (
            torch.bmm(demos, turnout_heads).squeeze(-1),
            torch.bmm(demos, partisanship_heads).squeeze(-1),
        )
        previous_partisanship = torch.tensor(
            [np.array(self.previous_partisanships[y]) for y in years]
        ).float()
        tp = tp + previous_partisanship * t
        return t, tp

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

    def predict(self, year, features, **kwargs):
        t, tp = self([year], [features], **kwargs)
        return (tp / t).detach().numpy()[0], t.detach().numpy()[0]

    @staticmethod
    def train(
        years,
        features,
        previous_partisanships,
        target_turnouts,
        target_partisanships,
        cvaps,
        iters=1000,
        lr=1e-2,
        *,
        dimensions,
    ):
        torch.manual_seed(0)
        if dimensions is None:
            # TODO check
            dimensions = features[0].shape[1] - 1
        dcm = DemographicCategoryPredictor(dimensions + 1, 10, years, previous_partisanships)
        dcm = train_torch_model(
            dcm,
            iters,
            lr,
            years,
            features,
            target_turnouts,
            target_partisanships,
            cvaps,
        )
        return dcm


@permacache(
    "2024bot/torch_model/train_torch_model",
    key_function=dict(
        dcm=hash_model,
        args=lambda args: [stable_hash(np.array(x)) for x in args],
    ),
)
def train_torch_model(dcm, iters, lr, *args):
    opt = torch.optim.Adam(dcm.parameters(), lr=lr)
    for itr in range(iters):
        opt.zero_grad()
        lv = dcm.loss(*args)
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
    partisanship_noise = attr.ib(default=0)
    turnout_noise = attr.ib(default=0)
    turnout_weights = attr.ib(default=None)

    @staticmethod
    def train(*, years, features, data, feature_kwargs):
        turnouts = {y: data[y].total_votes / data[y].CVAP for y in years}
        dcm = DemographicCategoryPredictor.train(
            years,
            features=[features.features(y) for y in years],
            previous_partisanships={y : np.array(data[y].past_pres_partisanship) for y in years},
            target_turnouts=[turnouts[y] for y in years],
            target_partisanships=[data[y].dem_margin for y in years],
            cvaps=[data[y].CVAP for y in years],
            iters=6000,
            **feature_kwargs,
        )
        residuals = {}
        for y in years:
            p, t = dcm.predict(y, features.features(y))
            residuals[y] = data[y].dem_margin - p, turnouts[y] - t
        return AdjustedDemographicCategoryModel(dcm, residuals, StableTrendModel(0))

    def perturb(self, *, prediction_seed, alpha_partisanship, alpha_turnout):
        if prediction_seed is None:
            return self
        torch.manual_seed(prediction_seed)
        partisanship_noise = (torch.randn(self.dcm.d) * alpha_partisanship).float()
        turnout_noise = (torch.randn(self.dcm.d) * alpha_turnout).float()
        turnout_weights = torch.randn(len(self.dcm.years)).float()
        turnout_weights /= turnout_weights.sum()

        return AdjustedDemographicCategoryModel(
            dcm=self.dcm,
            residuals=self.residuals,
            trend_model=self.trend_model,
            partisanship_noise=partisanship_noise,
            turnout_noise=turnout_noise,
            turnout_weights=turnout_weights,
        )

    def predict(self, *, model_year, year, features, correct):
        turnout_weights = self.turnout_weights
        if turnout_weights is None and model_year != year:
            turnout_weights = torch.tensor(
                [1/len(self.dcm.years)] * len(self.dcm.years)
            )
        p, t = self.dcm.predict(
            model_year,
            features,
            partisanship_noise=self.partisanship_noise,
            turnout_noise=self.turnout_noise,
            turnout_weights=turnout_weights,
        )
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
        adcm = self.adcm.perturb(prediction_seed=prediction_seed, alpha_partisanship=self.alpha, alpha_turnout=self.alpha * 0.5)
        model_year = 2020 if year == 2024 else year
        return self.adcm.predict(
            model_year=model_year,
            year=year,
            features=self.features.features(year),
            correct=correct,
        )
