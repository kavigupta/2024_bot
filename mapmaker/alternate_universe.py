import copy
from collections import Counter
import pickle
from mapmaker.colors import Profile

import numpy as np

from permacache import permacache

import requests
import torch
import torch.nn as nn


from mapmaker.aggregation import get_popular_vote, get_electoral_vote, get_state_results
from mapmaker.data import data_by_year
from mapmaker.stitch_map import generate_map

POP_VOTE_SIGMA = 5e-2
POP_VOTE_PRECISION = 0.1
MAX_EC_WIN = 538


def generate_alternate_universe_map(seed, title, path):
    from mapmaker.generate_image import get_model

    model = get_model(calibrated=False)
    copied_model = copy.deepcopy(model)
    pv_seed, torch_seed, symbol_seed, color_seed = np.random.RandomState(seed).choice(
        2 ** 32, size=4
    )
    torch.manual_seed(torch_seed)
    target_popular_vote = np.random.RandomState(pv_seed).randn() * POP_VOTE_SIGMA
    while True:
        copied_model.adcm.dcm.turnout_heads["2020"] = nn.Parameter(
            torch.randn(copied_model.adcm.dcm.turnout_heads["2020"].shape)
        )
        copied_model.adcm.dcm.partisanship_heads["(2020, False)"] = nn.Parameter(
            torch.randn(copied_model.adcm.dcm.partisanship_heads["(2020, False)"].shape)
        )
        p, t = copied_model.adcm.dcm.predict(
            2020, copied_model.features.features(2020), use_past_partisanship=False
        )

        popular_vote = get_popular_vote(data_by_year()[2020], dem_margin=p, turnout=t)
        print(popular_vote, target_popular_vote)
        if abs(target_popular_vote - popular_vote) > POP_VOTE_PRECISION:
            continue
        if (
            max(get_electoral_vote(data_by_year()[2020], dem_margin=p, turnout=t))
            > MAX_EC_WIN
        ):
            continue
        break
    with open(path.replace(".svg", ".pkl"), "wb") as f:
        pickle.dump(get_state_results(data_by_year()[2020], turnout=t, dem_margin=p), f)
    generate_map(
        data_by_year()[2020],
        title=title,
        out_path=path,
        dem_margin=p,
        turnout=t,
        map_type="president",
        year=2020,
        profile=Profile(
            symbol=sample_symbols(symbol_seed),
            hue=sample_colors(color_seed),
            bot_name="bot_althistory",
        ),
    )


@permacache("2024bot/alternate_universe/character_frequencies")
def character_frequencies():
    results = (
        requests.get(
            "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"
        )
        .content.decode("utf-8")
        .split("\n")
    )
    results = [
        x
        for x in results
        if x and set(x) - {chr(c) for c in range(ord("a"), 1 + ord("z"))} == set()
    ]
    results = Counter(x[0] for x in results)
    return dict(results.items())


def sample_symbols(seed):
    frequencies = character_frequencies()
    letters = sorted(frequencies)
    weights = np.array([frequencies[c] for c in letters], dtype=np.double)
    weights /= weights.sum()
    rng = np.random.RandomState(seed)
    while True:
        a, b = rng.choice(letters, size=2, p=weights)
        if a != b:
            break
    return dict(dem=a.upper(), gop=b.upper())


def distance(a, b):
    a, b = project(a), project(b)
    if a > b:
        a, b = b, a
    return min(abs(a - b), abs(a + 1 - b))


def project(x):
    if x < 0.5:
        # red to green gets compressed
        x = 0.25 + x / 2
    # stretch back to the whole circle
    x = (x - 0.25) / 0.75
    return x


def sample_colors(seed):
    rng = np.random.RandomState(seed)
    while True:
        a, b = rng.rand(2)
        if distance(a, b) > 1 / 3:
            break
    return dict(dem=a, gop=b)
