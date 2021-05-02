import os
from functools import lru_cache

from urllib.request import urlopen
import json

import numpy as np
import pandas as pd

CSVS = os.path.join(os.path.dirname(__file__), "../csvs")


@lru_cache(None)
def counties():
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        c = json.load(response)
    [lakota] = [x for x in c["features"] if x["id"] == "46113"]
    lakota["id"] = "46102"
    return c


def data_for_year(year):
    data = pd.read_csv(
        f"{CSVS}/election_demographic_data - {year}.csv", dtype=dict(FIPS=str)
    )
    data["FIPS"] = data["FIPS"].map(lambda x: x if len(x) == 5 else "0" + x)
    return data[
        [
            "FIPS",
            "state",
            "county",
            "total_votes",
            "dem_margin",
            "past_pres_partisanship",
            "CVAP",
            "median_age",
            "bachelor %",
            "median_income",
            "rural %",
            "white %",
            "black %",
            "native %",
            "asian %",
            "hispanic %",
            "evangelical",
            "protestant",
            "catholic",
            "mormon",
            "total_religious",
            "black_protestant",
            "poverty",
        ]
    ]


@lru_cache(None)
def all_data(year):

    if year != 2024:
        all_data = data_for_year(year)
    else:
        all_data = data_for_year(2020)
        data_2016 = data_for_year(2016)
        data_2012 = data_for_year(2012)
        all_data["past_pres_partisanship"] = all_data["dem_margin"]
        all_data["dem_margin"] = 0.
        keys = ["CVAP", "median_age", "bachelor %", "median_income", "white %", 
                "black %", "native %", "asian %", "hispanic %", "poverty"]
        for key in keys:
            all_data[key] = all_data[key] + ((all_data[key] - data_2016[key])) 
            #* 2./3 + ((all_data[key] - data_2012[key])) * 1./3

    ## Nonlinearity
    all_data["county_diversity_black_white"] = all_data["black %"] * all_data["white %"]
    all_data["county_diversity_hispanic_white"] = (
        all_data["hispanic %"] * all_data["white %"]
    )
    all_data["county_diversity_white_homogenity"] = all_data["white %"] ** 2
    all_data["county_diversity_white_education"] = (
        all_data["white %"] ** 2 * all_data["bachelor %"]
    )
    all_data["county_diversity_hispanic_homogenity"] = all_data["hispanic %"] ** 2
    all_data["county_diversity_native_homogenity"] = all_data["native %"] ** 2

    # all_data["turnout_spike"] = np.clip(
    #     all_data["2018 votes"] / all_data["2016_votes"], 0, 3
    # )
    all_data["hispanic_rural"] = all_data["hispanic %"] ** 2 * all_data["rural %"]

    all_data["turnout"] = all_data["total_votes"] / all_data["CVAP"]

    # Poverty Nonlinearities
    all_data["poverty_black_nonlinearity"] = (
        all_data["black %"] ** 2 * all_data["poverty"]
    )
    all_data["poverty_white_nonlinearity"] = (
        all_data["white %"] ** 2 * all_data["poverty"]
    )

    def logify(column):
        all_data[column] = np.log(all_data[column]).replace(-np.inf, -1000)

    logify("median_income")
    all_data["population"] = all_data["CVAP"]
    logify("population")

    if year == 2020:
        all_data["biden_2020"] = all_data["dem_margin"]
    # logify("2012votes")
    # logify("2016_votes")
    # logify("2018 votes")
    # logify("medianincome_2012")

    return all_data


def data_by_year():
    return {
        year: all_data(year)
        for year in (2012, 2016, 2020, 2024)
    }


def ec():
    return pd.read_csv(os.path.join(CSVS, "ec.csv")).set_index("state")
