from abc import ABC, abstractmethod

import numpy as np

from .aggregation import get_electoral_vote, get_state_results, get_popular_vote
from .stitch_map import generate_map


class Model(ABC):
    @abstractmethod
    def fully_random_sample(self, *, year, prediction_seed, correct):
        pass

    @property
    @abstractmethod
    def metadata(self):
        pass

    def family_of_predictions(self, *, year, correct=True, n_seeds=1000):
        county_results, state_results, pop_votes = [], [], []
        for seed in range(n_seeds):
            predictions, turnout = self.fully_random_sample(
                year=year, correct=correct, prediction_seed=seed
            )
            county_results.append(predictions)
            state_results.append(
                get_state_results(
                    self.metadata, dem_margin=predictions, turnout=turnout
                )
            )
            pop_votes.append(
                get_popular_vote(self.metadata, dem_margin=predictions, turnout=turnout)
            )
        return np.array(county_results), np.array(state_results), np.array(pop_votes)

    def win_consistent_with(self, predictions, turnout, seed):
        if seed is None:
            return True
        dem, gop = get_electoral_vote(
            self.metadata, dem_margin=predictions, turnout=turnout
        )
        dem_win = dem > gop  # ties go to gop
        # even days, democrat. odd days, gop
        return dem_win == (seed % 2 == 0)

    def sample(self, *, year, seed=None, correct=True):
        rng = np.random.RandomState(seed)
        while True:
            predictions, turnout = self.fully_random_sample(
                year=year,
                prediction_seed=rng.randint(2 ** 32) if seed is not None else None,
                correct=correct,
            )
            if self.win_consistent_with(predictions, turnout, seed):
                break
        return predictions, turnout

    def sample_map(self, title, path, **kwargs):
        print(f"Generating {title}")
        predictions, turnout = self.sample(**kwargs)
        return generate_map(
            self.metadata,
            title,
            path,
            dem_margin=predictions,
            turnout=turnout,
        )
