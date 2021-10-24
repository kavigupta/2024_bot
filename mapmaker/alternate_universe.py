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
from mapmaker.stitch_map import produce_entire_map

POP_VOTE_SIGMA = 5e-2
POP_VOTE_PRECISION = 0.1e-2
MAX_EC_WIN = 538

TURNOUT_NOISE = 20
PARTISAN_NOISE = 5


def generate_alternate_universe_map(seed, title, path):
    from mapmaker.generate_image import get_model

    model = get_model(calibrated=False, num_demographics=30)
    copied_model = copy.deepcopy(model)
    pv_seed, torch_seed, symbol_seed, color_seed = np.random.RandomState(seed).choice(
        2 ** 32, size=4
    )
    torch.manual_seed(torch_seed)
    target_popular_vote = np.random.RandomState(pv_seed).randn() * POP_VOTE_SIGMA
    while True:
        copied_model.adcm.dcm.turnout_heads["2020"] = nn.Parameter(
            TURNOUT_NOISE
            * torch.randn(copied_model.adcm.dcm.turnout_heads["2020"].shape)
        )
        copied_model.adcm.dcm.partisanship_heads["(2020, False)"] = nn.Parameter(
            PARTISAN_NOISE
            * torch.randn(
                copied_model.adcm.dcm.partisanship_heads["(2020, False)"].shape
            )
        )
        p, t = copied_model.adcm.dcm.predict(
            2020, copied_model.features.features(2020), use_past_partisanship=False
        )

        popular_vote = get_popular_vote(data_by_year()[2020], dem_margin=p, turnout=t)
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

    names = sample_party_names(symbol_seed)
    produce_entire_map(
        data_by_year()[2020],
        title=title,
        out_path=path,
        dem_margin=p,
        turnout=t,
        map_type="president",
        year=2020,
        profile=Profile(
            symbol={k: v[0] for k, v in names.items()},
            name=names,
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


def sample_party_names(seed):
    weights, names = zip(*party_names())
    weights = np.array(weights, dtype=np.float)
    weights /= weights.sum()
    rng = np.random.RandomState(seed)
    while True:
        a, b = rng.choice(len(names), size=2, p=weights)
        a, b = names[a], names[b]
        if a[0] != b[0]:
            break
    print(a, b)
    return dict(dem=a, gop=b)


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


def sample_colors(seed):
    rng = np.random.RandomState(seed)
    while True:
        a, b = [sample_color(rng) for _ in range(2)]
        if distance(a, b) > 1 / 4:
            break
    return dict(dem=a, gop=b)


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
