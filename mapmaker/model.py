from abc import ABC, abstractmethod

import copy

import numpy as np

from .aggregation import get_electoral_vote, get_state_results, get_popular_vote
from .stitch_map import generate_map

from .features import Features


class Model(ABC):
    def __init__(self, data_by_year, feature_kwargs):
        self.data = data_by_year
        self.features = Features.fit(data_by_year, train_key=2020, **feature_kwargs)
        self.alpha = 0
        self.dekurt_param = 0

    @abstractmethod
    def fully_random_sample(
        self, *, year, prediction_seed, correct, turnout_year, map_type
    ):
        pass

    def with_alpha(self, alpha):
        self = copy.copy(self)
        self.alpha = alpha
        return self

    def with_dekurt(self, dekurt):
        self = copy.copy(self)
        self.dekurt_param = dekurt
        return self

    def family_of_predictions(self, *, year, correct=True, n_seeds=1000):
        county_results, state_results, pop_votes = [], [], []
        for seed in range(n_seeds):
            predictions, turnout = self.fully_random_sample(
                year=year,
                correct=correct,
                prediction_seed=seed,
                turnout_year=None,
                map_type="president",
            )
            county_results.append(predictions)
            state_results.append(
                get_state_results(
                    self.data[year], dem_margin=predictions, turnout=turnout
                )
            )
            pop_votes.append(
                get_popular_vote(
                    self.data[year], dem_margin=predictions, turnout=turnout
                )
            )
        return np.array(county_results), np.array(state_results), np.array(pop_votes)

    def win_consistent_with(self, predictions, turnout, seed, *, year):
        if seed is None:
            return True
        dem, gop = get_electoral_vote(
            self.data[year], dem_margin=predictions, turnout=turnout
        )
        dem_win = dem > gop  # ties go to gop
        # even days, democrat. odd days, gop
        return dem_win == (seed % 2 == 0)

    def sample(self, *, year, seed=None, correct=True, turnout_year=None, map_type):
        rng = np.random.RandomState(seed)
        while True:
            predictions, turnout = self.fully_random_sample(
                year=year,
                prediction_seed=rng.randint(2 ** 32) if seed is not None else None,
                correct=correct,
                turnout_year=turnout_year,
                map_type=map_type,
            )
            if self.win_consistent_with(predictions, turnout, seed, year=year):
                break
        return predictions, turnout

    def sample_map(self, title, path, *, year, map_type, **kwargs):
        print(f"Generating {title}")
        predictions, turnout = self.sample(year=year, **kwargs, map_type=map_type)
        return generate_map(
            self.data[year],
            title,
            path,
            dem_margin=predictions,
            turnout=turnout,
            map_type=map_type,
            year=year,
        )
