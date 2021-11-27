import copy
import json

import tqdm.auto as tqdm
import numpy as np

from permacache import permacache, stable_hash

from ..aggregation import get_electoral_vote, get_popular_vote
from ..data import data_by_year
from ..generate_image import get_model
from ..mapper import USAPresidencyBaseMap
from ..version import version


@permacache("2024bot/scripts/sample_direct_sample2")
def sample_direct(seed, version=version):
    d = data_by_year()[2024]
    p, t = get_model(calibrated=True).sample(
        year=2024, seed=seed, correct=True, basemap=USAPresidencyBaseMap()
    )
    dem_ec = get_electoral_vote(
        d, dem_margin=p, turnout=t, basemap=USAPresidencyBaseMap()
    )
    pop_vote = get_popular_vote(d, dem_margin=p, turnout=t)
    return p, dem_ec, pop_vote


def sample_balanced(count):
    d = data_by_year()[2024]
    ps, ecs, pvs = [], [], []
    for seed in tqdm.tqdm(np.random.RandomState(0).choice(2 ** 32, size=count)):
        p, ec, pv = sample_direct(seed)
        ps.append(p)
        ecs.append(ec)
        pvs.append(pv)
    ps = np.array(ps)
    ecs = np.array(ecs)
    pvs = np.array(pvs)

    dem_wins = ecs[:, 0] >= 270
    dem_win_idxs, gop_win_idxs = np.where(dem_wins)[0], np.where(~dem_wins)[0]
    if dem_win_idxs.size > gop_win_idxs.size:
        dem_win_idxs = np.random.RandomState(0).choice(
            dem_win_idxs, size=gop_win_idxs.shape, replace=False
        )
    else:
        gop_win_idxs = np.random.RandomState(0).choice(
            gop_win_idxs, size=dem_win_idxs.shape, replace=False
        )
    idxs = [*gop_win_idxs, *dem_win_idxs]
    ps = ps[idxs]
    ecs = ecs[idxs]
    print((ecs[:, 0] >= 270).mean())
    pvs = pvs[idxs]
    return ps, ecs, pvs


@permacache(
    "2024bot/scripts/compute_bellweather_2",
    key_function=dict(ps=stable_hash, ecs=stable_hash),
)
def compute_bellweather(ps, ecs):
    values = np.linspace(-1, 1, 1001)
    belweather = []
    for county_idx in tqdm.trange(ps.shape[1]):
        belweather.append(
            ((ps[:, county_idx, None] > values) == (ecs[:, 0] >= 270)[:, None]).mean(0)
        )
    belweather = np.array(belweather)
    by_best_divider = belweather.max(1)
    assert values[belweather.shape[1] // 2] == 0
    by_even = belweather[:, belweather.shape[1] // 2]
    return by_even, by_best_divider


def annotate_geojson(by_even, by_best_divider, path):

    d = data_by_year()[2024]
    c = copy.deepcopy(USAPresidencyBaseMap().counties)
    c["features"] = [
        feat
        for feat in c["features"]
        if feat["id"][:2] != "72" and feat["id"] != "15005"
    ]
    reverse_map = {fips: idx for idx, fips in enumerate(d.FIPS)}
    for feat in c["features"]:
        feat["properties"]["by_best_divider"] = (
            by_best_divider[reverse_map[feat["id"]]] * 100
        )
        feat["properties"]["by_even"] = by_even[reverse_map[feat["id"]]] * 100
    with open(path, "w") as f:
        json.dump(c, f)


def output(path):
    ps, ecs, pvs = sample_balanced(10_000)
    by_even, by_best_divider = compute_bellweather(ps, ecs)
    annotate_geojson(by_even, by_best_divider, path)
