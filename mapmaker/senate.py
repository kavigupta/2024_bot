import zlib
from functools import lru_cache
import numpy as np

class_1_and_3 = {
    "WA",
    "ND",
    "CA",
    "NV",
    "UT",
    "AZ",
    "HI",
    "MO",
    "WI",
    "IN",
    "OH",
    "PA",
    "NY",
    "VT",
    "CT",
    "MD",
    "FL",
}

class_2_and_3 = {
    "AK",
    "AL",
    "OR",
    "ID",
    "CO",
    "SD",
    "KS",
    "OK",
    "IA",
    "AR",
    "LA",
    "IL",
    "KY",
    "SC",
    "GA",
    "NC",
    "NH",
    "NE",
}

class_1_and_2 = {
    "MT",
    "UT",
    "NM",
    "WY",
    "TX",
    "MN",
    "MI",
    "TN",
    "MS",
    "WV",
    "VA",
    "NJ",
    "RI",
    "MA",
    "ME",
    "DE",
}

senate_2022 = class_1_and_3 | class_2_and_3

r_incumbents_up_2022 = {
    "ID",
    "UT",
    "ND",
    "SD",
    "NE",
    "OK",
    "IA",
    "AR",
    "LA",
    "WI",
    "IN",
    "KY",
    "FL",
    "SC",
    "AK",
    "KS",
}

r_open_2022 = {
    "AL",
    "MO",
    "NC",
    "OH",
    "PA",
}

d_incumbents_up_2022 = {
    "CA",
    "OR",
    "WA",
    "NV",
    "AZ",
    "CO",
    "IL",
    "GA",
    "MD",
    "NY",
    "CT",
    "VT",
    "NH",
    "HI",
}

d_open_2022 = set()

r_lock_2022 = 50 - len(r_open_2022) - len(r_incumbents_up_2022)
d_lock_2022 = 50 - len(d_open_2022) - len(d_incumbents_up_2022)

assert (
    r_incumbents_up_2022 | d_incumbents_up_2022 | r_open_2022 | d_open_2022
) == senate_2022

# https://fivethirtyeight.com/features/how-much-was-incumbency-worth-in-2018/
INCUMBENCY_EFFECT_MEAN = 2.7e-2
# ????? complete guess for now
CANDIDATE_EFFECT_MEAN = 2e-2

@lru_cache(None)
def race_effect(state, seed):
    assert state not in d_incumbents_up_2022 or state not in r_incumbents_up_2022
    if seed is None:
        candidate_effect = 0
        incumbency_effect = INCUMBENCY_EFFECT_MEAN
    else:
        rng = np.random.RandomState((seed + zlib.adler32(state.encode("utf-8"))) % 2 ** 32)
        # mean of absolute value normal value == sqrt(2 / pi)
        candidate_effect = rng.randn() * (np.pi / 2) ** 0.5 * CANDIDATE_EFFECT_MEAN
        # chi squared mode = df - 2
        incumbency_effect = rng.chisquare(INCUMBENCY_EFFECT_MEAN + 2)

    effect = candidate_effect
    if state in d_incumbents_up_2022:
        effect += incumbency_effect
    elif state in r_incumbents_up_2022:
        effect -= incumbency_effect
    return effect
