from collections import defaultdict

import attr

import torch
import torch.nn as nn

import numpy as np

from permacache import permacache, stable_hash

from .aggregation import get_popular_vote
from .model import Model
from .trend_model import StableTrendModel, NoisedTrendModel
from .utils import hash_model, intersect_all

ITERS = 12000
DEMOS_SIMILARITY_LOSS_WEIGHT = 1

HIDDEN_SIZE = 100
LEARNING_RATE = 1e-2

YEAR_FORCE_ENVIRONMENT = {2022: 2e-2}
EXCLUDE_TURNOUT_YEARS = [2014]


class DemographicCategoryPredictor(nn.Module):
    # to refresh cache, increment this
    version = 4.8

    def __init__(self, f, d, years, previous_partisanships, fipses, gamma=0.5):
        super().__init__()
        self.f = f
        self.d = d
        self.gamma = gamma
        self.years = years
        self.min_turn = 0.4
        self.max_turn = 0.8
        self.version = self.version
        self.demos_similarity_loss_weight = DEMOS_SIMILARITY_LOSS_WEIGHT
        assert set(years) == set(previous_partisanships) == set(fipses)
        self.previous_partisanships = previous_partisanships
        common_fips = intersect_all(fipses.values())
        self.index_to_common = {y: np.in1d(fipses[y], common_fips) for y in years}
        # self.latent_demographic_model = nn.Sequential(nn.Linear(f, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, d), nn.Softmax(-1))
        self.latent_demographic_model = nn.Sequential(nn.Linear(f, d), nn.Softmax(-1))
        self.turnout_heads = nn.ParameterDict(
            {str(y): nn.Parameter(torch.randn(d, 1)) for y in years}
        )
        self.partisanship_heads = nn.ParameterDict(
            {
                str((y, use_past)): nn.Parameter(torch.randn(d, 1))
                for y in years
                for use_past in (True, False)
            }
        )

    def get_heads(
        self,
        y,
        *,
        use_past_partisanship,
        partisanship_noise=0,
        turnout_noise=0,
        turnout_weights=None,
        turnout_year=None,
    ):

        turnout = {
            year: self.create_turnout_head(
                self.turnout_heads[str(year)] + turnout_noise
            )
            for year in self.years
        }

        partisanship = torch.tanh(
            self.partisanship_heads[str((y, use_past_partisanship))]
            + partisanship_noise
        )

        if turnout_weights is not None:
            assert turnout_year is None
            turnout = sum(
                [turnout[year] * turnout_weights[year] for year in turnout_weights]
            )
        else:
            turnout_year = y if turnout_year is None else turnout_year
            turnout = turnout[turnout_year]

        return turnout, turnout * partisanship

    def create_turnout_head(self, head):
        return torch.sigmoid(head) * (self.max_turn - self.min_turn) + self.min_turn

    def forward(
        self, features, full_output=False, use_past_partisanship=True, **kwargs
    ):
        years = list(features)
        features = {y: torch.tensor(features[y]).float() for y in features}
        demos = {y: self.latent_demographic_model(features[y]) for y in features}
        heads = {
            y: self.get_heads(y, **kwargs, use_past_partisanship=use_past_partisanship)
            for y in years
        }
        t, tp = {}, {}
        for y in years:
            turnout_heads, partisanship_heads = heads[y]
            t[y] = (demos[y] @ turnout_heads).squeeze(-1)
            tp[y] = (demos[y] @ partisanship_heads).squeeze(-1)
            previous_partisanship = torch.tensor(
                np.array(self.previous_partisanships[y])
            ).float()
            # tp[y] = tp[y] + previous_partisanship * t[y]
            if use_past_partisanship:
                tp[y] = (
                    torch.tanh(
                        torch.atanh(tp[y] / t[y]) + torch.atanh(previous_partisanship)
                    )
                    * t[y]
                )
        if full_output:
            return t, tp, demos
        return t, tp

    def loss(
        self,
        features,
        target_turnouts,
        target_partisanships,
        cvaps,
        *,
        use_past_partisanship,
    ):
        assert (
            target_turnouts.keys()
            == target_partisanships.keys()
            == features.keys()
            == cvaps.keys()
        )

        years = sorted(features.keys())
        target_turnouts = {y: np.array(target_turnouts[y]) for y in years}
        target_partisanships = {y: np.array(target_partisanships[y]) for y in years}
        target_tp = {y: target_turnouts[y] * target_partisanships[y] for y in years}
        target_t = {y: torch.tensor(target_turnouts[y]).float() for y in years}
        target_tp = {y: torch.tensor(target_tp[y]).float() for y in years}
        cvaps = {y: torch.tensor(np.array(cvaps[y])).float() for y in years}
        t, tp, demos = self(
            features, full_output=True, use_past_partisanship=use_past_partisanship
        )

        demos = torch.stack([demos[y][self.index_to_common[y]] for y in years])
        cvaps_common = cvaps[2020][self.index_to_common[2020]]
        demos_similarity_loss = (
            demos.var(0) * cvaps_common[:, None]
        ).sum() / cvaps_common.sum()
        losses = []
        for y in years:
            loss = (target_t[y] - t[y]) ** 2 * self.gamma + (target_tp[y] - tp[y]) ** 2
            losses.append((loss * cvaps[y]).sum() / cvaps[y].sum())
        return (
            sum(losses) / len(losses),
            demos_similarity_loss * self.demos_similarity_loss_weight,
        )

    def predict(self, year, features, **kwargs):
        t, tp = self({year: features}, **kwargs)
        t, tp = t[year], tp[year]
        return (tp / t).detach().numpy(), t.detach().numpy()

    def get_demographics_by_county(self, features):
        features = torch.tensor(features).float()
        return self.latent_demographic_model(features).detach().numpy()

    @staticmethod
    def train(
        features,
        previous_partisanships,
        fipses,
        target_turnouts,
        target_partisanships,
        cvaps,
        iters=1000,
        *,
        dimensions,
        num_demographics,
    ):
        torch.manual_seed(0)
        if dimensions is None:
            dimensions = features[list(features)[0]].shape[1] - 1
        dcm = DemographicCategoryPredictor(
            dimensions + 1,
            num_demographics,
            list(target_turnouts),
            previous_partisanships,
            fipses,
        )
        dcm = train_torch_model(
            dcm,
            iters,
            LEARNING_RATE,
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
        args=lambda args: [
            {y: stable_hash(np.array(x)) for y, x in xs.items()} for xs in args
        ],
    ),
)
def train_torch_model(dcm, iters, lr, *args):
    opt = torch.optim.Adam(dcm.parameters(), lr=lr)
    for itr in range(iters):
        for use_past_partisanship in True, False:
            opt.zero_grad()
            lv, lvdemo = dcm.loss(*args, use_past_partisanship=use_past_partisanship)
            if (itr + 1) % 100 == 0:
                print(f"{itr:4d} {lv.item():.6f} {lvdemo.item():.6f}")
            (lv + lvdemo).backward()
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
    def train(*, years, features, data, feature_kwargs, num_demographics):
        turnouts = {y: data[y].total_votes / data[y].CVAP for y in years}
        dcm = DemographicCategoryPredictor.train(
            features={y: features.features(y) for y in years},
            previous_partisanships={
                y: np.array(data[y].past_pres_partisanship) for y in years
            },
            fipses={y: np.array(data[y].FIPS) for y in years},
            target_turnouts={y: turnouts[y] for y in years},
            target_partisanships={y: data[y].dem_margin for y in years},
            cvaps={y: data[y].CVAP for y in years},
            iters=ITERS,
            **feature_kwargs,
            num_demographics=num_demographics,
        )
        residuals = {}
        for y in years:
            p, t = dcm.predict(y, features.features(y), turnout_year=None)
            residuals[y] = data[y].dem_margin - p, turnouts[y] - t
        return AdjustedDemographicCategoryModel(dcm, residuals, StableTrendModel(0))

    def perturb(self, *, for_year, prediction_seed, alpha_partisanship, alpha_turnout):
        if prediction_seed is None:
            return self
        rng = np.random.RandomState(prediction_seed)
        torch.manual_seed(rng.randint(2 ** 32))
        partisanship_noise = (self.sample_perturbations() * alpha_partisanship).float()
        turnout_noise = (self.sample_perturbations() * alpha_turnout).float()
        same_cycle_years = self.turnout_relevant_years(for_year)
        turnout_weights = torch.rand(len(same_cycle_years)).float()
        turnout_weights /= turnout_weights.sum()
        turnout_weights = {y: w for y, w in zip(same_cycle_years, turnout_weights)}
        trend_model = NoisedTrendModel.of(rng, self.dcm.f)

        return AdjustedDemographicCategoryModel(
            dcm=self.dcm,
            residuals=self.residuals,
            trend_model=trend_model,
            partisanship_noise=partisanship_noise,
            turnout_noise=turnout_noise,
            turnout_weights=turnout_weights,
        )

    def sample_perturbations(self):
        deltas = torch.rand(self.dcm.d, 1) - 0.5
        return deltas

    def turnout_relevant_years(self, output_year):
        return [
            y
            for y in self.dcm.years
            if y % 4 == output_year % 4 and y not in EXCLUDE_TURNOUT_YEARS
        ]

    def predict(
        self,
        *,
        data,
        model_year,
        output_year,
        features,
        correct,
        turnout_year,
        basemap,
    ):
        turnout_weights = self.turnout_weights
        if turnout_weights is None and model_year != output_year:
            same_cycle_years = self.turnout_relevant_years(output_year)
            turnout_weights = {y: 1 / len(same_cycle_years) for y in same_cycle_years}
        p, t = self.dcm.predict(
            model_year,
            features,
            partisanship_noise=self.partisanship_noise,
            turnout_noise=self.turnout_noise,
            turnout_weights=turnout_weights,
            turnout_year=turnout_year,
        )
        if correct:
            pr = self.trend_model(
                features,
                self.residuals[model_year][0],
                year=output_year,
                base_year=model_year,
            )
            if turnout_weights is not None:
                for ty, tw in turnout_weights.items():
                    tw = float(tw)
                    # complete mess. Map both sets of indices to a common basis and add the residuals at that site
                    # Just default to residual 0 for ones we we don't have direct access to
                    t[self.dcm.index_to_common[model_year]] += (
                        self.residuals[ty][1] * tw
                    )[self.dcm.index_to_common[ty]]
            else:
                t = t + self.residuals[turnout_year or model_year][1]
            if correct == "just_residuals":
                p = pr
            else:
                p = p + pr
                if output_year in YEAR_FORCE_ENVIRONMENT:
                    empirical = get_popular_vote(data, dem_margin=p, turnout=t)
                    p = p - empirical + YEAR_FORCE_ENVIRONMENT[output_year]

        return np.clip(p, -0.99, 0.99), np.clip(t, 0.01, 0.99)


class DemographicCategoryModel(Model):
    def __init__(self, data_by_year, feature_kwargs={}, *, num_demographics):
        super().__init__(data_by_year, feature_kwargs)
        self.adcm = AdjustedDemographicCategoryModel.train(
            years=sorted(y for y in data_by_year if y <= 2020),
            features=self.features,
            data=self.data,
            feature_kwargs=feature_kwargs,
            num_demographics=num_demographics,
        )

    def fully_random_sample(
        self, *, year, prediction_seed, correct, turnout_year, basemap
    ):
        # use the 2020 predictor since that's the best we have
        # TODO ADD THE PERTURBATIONS
        adcm = self.adcm.perturb(
            for_year=year,
            prediction_seed=prediction_seed,
            alpha_partisanship=self.alpha,
            alpha_turnout=self.alpha * 10,
        )
        model_year = 2020 if year > 2020 else year
        return adcm.predict(
            data=self.data[year],
            model_year=model_year,
            output_year=year,
            features=self.features.features(year),
            correct=correct,
            turnout_year=turnout_year,
            basemap=basemap,
        )

    def get_demographics_by_county(self, *, year):
        return self.adcm.dcm.get_demographics_by_county(self.features.features(year))
