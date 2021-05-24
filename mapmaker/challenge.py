import numpy as np

from electiondata.examples.nytimes_num_colleges import NYTimesNumColleges

from .aggregation import get_popular_vote
from .data import data_by_year
from .stitch_map import generate_map


def directional_margin(data, statistic, reverse=False):
    turnout = data.total_votes / data.CVAP
    sorts = np.argsort(statistic)
    if reverse:
        sorts = sorts[::-1]
    ordering = np.argsort(sorts)
    dem_margin = np.array(sorted(data.dem_margin))[np.array(ordering)]

    correction = get_popular_vote(
        data, dem_margin=data.dem_margin, turnout=turnout
    ) - get_popular_vote(data, dem_margin=dem_margin, turnout=turnout)
    return dem_margin + correction


def get_margin_auto(data, statistic):
    correlation = np.corrcoef(directional_margin(data, statistic), data.dem_margin)[
        0, 1
    ]
    if correlation > 0:
        print("higher = more dem")
        return directional_margin(data, statistic)
    print("lower = more dem")
    return directional_margin(data, statistic, reverse=True)


def generate_challenge_maps(i, title, extractor):
    data = data_by_year()[2020]
    valid_fips = set(data.FIPS)
    college = NYTimesNumColleges().get().rename(columns={"county_fips": "FIPS"})
    college = college[college.FIPS.apply(lambda x: x in valid_fips)]
    data = data.merge(college, how="outer").fillna(0)
    turnout = data.total_votes / data.CVAP
    dem_margin = get_margin_auto(data, extractor(data))
    for is_solution in True, False:
        generate_map(
            data,
            f"Challenge {i}" + (": " + title) * is_solution,
            f"challenges/{i}" + "_solution" * is_solution + ".svg",
            dem_margin=dem_margin,
            turnout=turnout,
        )
