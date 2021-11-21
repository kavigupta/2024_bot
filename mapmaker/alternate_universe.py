import copy
from collections import Counter
import pickle

from pandas.core.frame import DataFrame
from mapmaker.colors import Profile

import numpy as np

from permacache import permacache

import requests
import torch
import torch.nn as nn


from mapmaker.aggregation import (
    get_electoral_vote,
    get_electoral_vote_by_voteshare,
    get_popular_vote_by_voteshare,
    get_state_results,
    get_state_results_by_voteshare,
    to_winning_margin,
    to_winning_margin_single,
)
from mapmaker.colors import DEFAULT_CREDIT
from mapmaker.data import data_by_year
from mapmaker.mapper import USAPresidencyBaseMap
from mapmaker.stitch_map import (
    produce_entire_map,
    produce_entire_map_generic,
    serialize_output,
)

POP_VOTE_SIGMA = 5e-2
POP_VOTE_PRECISION = 0.1e-2
MAX_EC_WIN = 538

TURNOUT_NOISE = 20
PARTISAN_NOISE = 5


def generate_alternate_universe_map(seed, title, path):
    from mapmaker.generate_image import get_model

    basemap = USAPresidencyBaseMap()

    data = data_by_year()[2020]

    model = get_model(calibrated=False, num_demographics=30)
    copied_model = copy.deepcopy(model)
    pv_seed, torch_seed, profile_seed = np.random.RandomState(seed).choice(
        2 ** 32, size=3
    )
    profile = sample_profile(seed, profile_seed)
    torch.manual_seed(torch_seed)
    target_popular_vote = abs(np.random.RandomState(pv_seed).randn() * POP_VOTE_SIGMA)
    while True:
        turnout = copied_model.adcm.dcm.create_turnout_head(
            TURNOUT_NOISE
            * torch.randn(copied_model.adcm.dcm.turnout_heads["2020"].shape)
        )

        partisanship = (
            PARTISAN_NOISE * torch.randn(turnout.shape[0], len(profile.name))
        ).softmax(-1)
        demos = copied_model.adcm.dcm.latent_demographic_model(
            torch.tensor(copied_model.features.features(2020)).float()
        )
        t = demos @ turnout
        tp = demos @ (turnout * partisanship)
        p = tp / t
        p = p.detach().numpy()
        t = t.detach().numpy()[:, 0]

        voteshare_by_party = dict(zip(sorted(profile.name), p.T))

        popular_vote = get_popular_vote_by_voteshare(
            data, voteshare_by_party=voteshare_by_party, turnout=t
        )

        _, popular_vote_margin = to_winning_margin_single(popular_vote)

        if abs(target_popular_vote - popular_vote_margin) > POP_VOTE_PRECISION:
            continue

        winner_by_county = to_winning_margin(voteshare_by_party=voteshare_by_party)
        states_by_party = {
            party: {s for (p, _), s in zip(winner_by_county, data.state) if p == party}
            for party in voteshare_by_party
        }
        ec = get_electoral_vote_by_voteshare(
            data,
            voteshare_by_party=voteshare_by_party,
            turnout=t,
            basemap=basemap,
        )
        if min(len(v) for v in states_by_party.values()) < 8 and min(ec.values()) == 0:
            continue
        print(ec)
        if max(ec.values()) > MAX_EC_WIN:
            continue
        break
    with open(path.replace(".svg", ".pkl"), "wb") as f:
        pickle.dump(
            serialize_output(
                profile,
                get_state_results_by_voteshare(
                    data, turnout=t, voteshare_by_party=popular_vote
                ),
                always_whole=True,
            ),
            f,
        )

    produce_entire_map_generic(
        data_by_year()[2020],
        title=title,
        out_path=path,
        voteshare_by_party=voteshare_by_party,
        turnout=t,
        basemap=basemap,
        year=2020,
        profile=profile,
    )


def sample_profile(seed, profile_seed):
    symbol_seed, color_seed = np.random.RandomState(profile_seed).choice(
        2 ** 32, size=2
    )
    names = sample_party_names(seed, symbol_seed)
    return Profile(
        symbol={k: v[0] for k, v in names.items()},
        name=names,
        hue=sample_colors(names, color_seed),
        bot_name="bot_althistory",
        credit=DEFAULT_CREDIT,
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


def sample_party_names(which, seed):
    weights, names = zip(*party_names())
    weights = np.array(weights, dtype=np.float)
    weights /= weights.sum()
    rng = np.random.RandomState(seed)
    count = 2 if which % 2 == 0 else 3
    while True:
        name_idxs = rng.choice(len(names), size=count, p=weights)
        chosen_names = [names[i] for i in name_idxs]
        if len(set(n[0] for n in chosen_names)) == count:
            break
    return {name: name for name in chosen_names}


def distance(a, b):
    a, b = project(a), project(b)
    if a > b:
        a, b = b, a
    return min(abs(a - b), abs(a + 1 - b))


def project(x):
    if x < 1 / 3:
        # red to green gets compressed
        x = 1 / 6 + x / 2
    # stretch back to the whole circle
    x = (x - 1 / 6) / (5 / 6)
    return x


def sample_color(rng):
    # red, orange, yellow, green, blue, purple, magenta, red
    keypoints = np.array([0, 30, 60, 120, 240, 270, 360], dtype=np.float)
    keypoints = keypoints / 360
    segment = rng.choice(len(keypoints) - 1)
    a, b = keypoints[segment], keypoints[segment + 1]
    return rng.rand() * (b - a) + a


def sample_colors(names, seed):
    rng = np.random.RandomState(seed)
    while True:
        colors = [sample_color(rng) for _ in range(len(names))]
        if all(
            distance(colors[a], colors[b]) > 1 / 4
            for a in range(len(colors))
            for b in range(len(colors))
            if a < b
        ):
            break
    return {name: color for name, color in zip(sorted(names), colors)}


def party_names():
    parties = []

    def add(weight, *names):
        parties.extend([(weight, name) for name in names])

    # Major Current American Parties
    add(
        2,
        "Democratic",
        "Republican",
    )

    # Minor Current American Parties
    add(
        1.5,
        "Libertarian",
        "Green",
        "Patriot",
    )

    # Major Historical American Parties
    add(
        1.5,
        "Federalist",
        "Democratic-Republican",
        "Free Soil",
        "Whig",
        "Unionist",
        "Populist",
        "Socialist",
        "Progressive",
    )

    # Minor Historical American Parties
    add(
        1,
        "People's",
        "Farmer-Labor",
        "Workers",
        "America First",
        "Black Panther",
        "Patriot",
        "American",
    )

    # Common Foreign Parties
    add(
        1,
        "Labor",
        "Tory",
        "Liberal Democratic",
        "New Democratic",
        "Conservative",
        "Christian Democratic",
        "Social Democratic",
        "Alternative for America",
        "American National Congress",
        "Vox",
    )

    # Ideologies
    add(
        1,
        "Accelerationist",
        "Agrarianist",
        "Anarchist",
        "Ba'athist",
        "Bonapartist",
        "Caeserist",
        "Capitalist",
        "Communist",
        "Conservative",
        "Corporatist",
        "Decelerationist",
        "Dengist",
        "Distributist",
        "Environmentalist",
        "Fascist",
        "Feminist",
        "Georgist",
        "Globalist",
        "Humanist",
        "Integralist",
        "Juche",
        "Leninist",
        "Liberal",
        "Longist",
        "Luddite",
        "Luxembergist",
        "Maoist",
        "Marxist-Leninist",
        "Monarchist",
        "Multiculturalist",
        "National Socialist",
        "Nationalist",
        "Pacifist",
        "Pluralist",
        "Posadist",
        "Primitivist",
        "Revanchist",
        "Socialist",
        "Syndicalist",
        "Totalitarian",
        "Technocratic",
        "Titoist",
        "Transhumanist",
        "Urbanist",
    )

    return parties
