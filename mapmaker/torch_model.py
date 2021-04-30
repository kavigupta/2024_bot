import torch
import torch.nn as nn

import numpy as np

from .model import Model


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

    def loss(self, years, features, target_turnouts, target_partisanships):

        target_turnouts = np.array(target_turnouts)
        target_partisanships = np.array(target_partisanships)
        target_tp = target_turnouts * target_partisanships
        loss = nn.MSELoss()
        t, tp = self(years, features)
        return self.gamma * loss(t, torch.tensor(target_turnouts).float()) + loss(
            tp, torch.tensor(target_tp).float()
        )

    @staticmethod
    def train(
        years, features, target_turnouts, target_partisanships, iters=1000, lr=1e-2
    ):
        dcm = DemographicCategoryPredictor(20, 15, years)
        opt = torch.optim.Adam(dcm.parameters(), lr=lr)
        for itr in range(iters):
            opt.zero_grad()
            lv = dcm.loss(years, features, target_turnouts, target_partisanships)
            if (itr + 1) % 100 == 0:
                print(itr, lv.item())
            lv.backward()
            opt.step()
        return dcm


class DemographicCategoryModel(Model):
    def __init__(self, data_by_year, feature_kwargs={}):
        super().__init__(data_by_year, feature_kwargs)
        train_years = sorted(y for y in data_by_year if y <= 2020)
        train_years = [2012, 2020]
        # print((self.data[y].total_votes.fillna(0) / self.data[y].CVAP).max())
        self.dcm = DemographicCategoryPredictor.train(
            train_years,
            [self.features.features(y) for y in train_years],
            [self.data[y].total_votes.fillna(0) / self.data[y].CVAP for y in train_years],
            [self.data[y].dem_margin for y in train_years],
            iters=2_000
        )

    def fully_random_sample(self, *, year, prediction_seed, correct):
        # use the 2020 predictor since that's the best we have
        # TODO ADD THE PERTURBATIONS
        model_year = 2020 if year == 2024 else year
        t, tp = self.dcm([model_year], [self.features.features(year)])
        return (tp / t).detach().numpy()[0], t.detach().numpy()[0]
