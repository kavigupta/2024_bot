import os
from functools import lru_cache

import electiondata as e
from electiondata.examples.canonical_2018_general import Canonical2018General
from electiondata.examples.plotly_geojson import PlotlyGeoJSON

import numpy as np
import pandas as pd

CSVS = os.path.join(os.path.dirname(__file__), "../csvs")


@lru_cache(None)
def counties():
    return PlotlyGeoJSON(e.alaska.FOUR_REGIONS()).get()


def data_2018():
    data1 = Canonical2018General(
        alaska_handler=e.alaska.FOUR_REGIONS(),
        uncontested_replacements=["us state governor"],
        uncontested_replacement_mode="interpolate",
    ).get()
    data1 = e.Aggregator(
        grouped_columns=["county_fips"],
        aggregation_functions={"votes_other": sum, "votes_DEM": sum, "votes_GOP": sum},
        removed_columns=["district"],
    )(data1[(data1.office == "us house") & (~data1.special)])
    data1 = data1.rename(columns={"county_fips": "FIPS"})
    data1["total_votes"] = data1.votes_other + data1.votes_DEM + data1.votes_GOP
    data1["dem_margin"] = (data1.votes_DEM - data1.votes_GOP) / data1.total_votes
    data1 = data1[["FIPS", "total_votes", "dem_margin"]]
    data = data_for_year(2020)
    return data1.merge(
        data[[x for x in data if x not in {"total_votes", "dem_margin"}]], how="inner"
    )


def read_csv(path):
    data = pd.read_csv(path, dtype=dict(FIPS=str))
    data = data[[x for x in data if x != "Unnamed: 0"]]
    data_ak = data[data.state == "Alaska"].copy()
    data_not_ak = data[data.state != "Alaska"].copy()
    normalizer = e.usa_county_to_fips("state", alaska_handler=e.alaska.FOUR_REGIONS())
    normalizer.apply_to_df(data_ak, "county", "FIPS")
    data_ak.dem_margin *= data_ak.CVAP
    data_ak.past_pres_partisanship *= data_ak.CVAP

    firsts = ["gisjoin", "county"]
    sums = [
        "CVAP",
        "dem_margin",
        "past_pres_partisanship",
        "total_votes",
    ]
    means = [
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
    agg = e.Aggregator(
        grouped_columns=["FIPS"],
        aggregation_functions={
            **{k: lambda x: list(x)[0] for k in firsts},
            **{k: np.sum for k in sums},
            **{k: np.mean for k in means},
        },
    )
    data_ak = agg(data_ak)

    data_ak.dem_margin /= data_ak.CVAP
    data_ak.past_pres_partisanship /= data_ak.CVAP
    return pd.concat([data_ak, data_not_ak])


def data_for_year(year):
    if year == 2018:
        data = data_2018()
    else:
        data = read_csv(f"{CSVS}/election_demographic_data - {year}.csv")
    # delete nans
    data = data[data.dem_margin == data.dem_margin]
    if year % 4 == 2:
        # exclude uncontested races
        data = data[data.dem_margin.abs() < 0.95]
    data["FIPS"] = data["FIPS"].map(
        lambda x: x if len(x) == 5 or x.startswith("02") else "0" + x
    )
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

    if year == 2022:
        all_data = data_2024().copy()
    elif year != 2024:
        all_data = data_for_year(year)
    else:
        all_data = data_2024().copy()


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

    return all_data


def data_2024():
    all_data = data_for_year(2020)
    data_2016 = data_for_year(2016)
    data_2012 = data_for_year(2012)
    print("2012", data_2012["CVAP"].sum())
    print("2016", data_2016["CVAP"].sum())
    print("2020", all_data["CVAP"].sum())
    all_data["dem_margin"] = 0
    keys = [
        "median_age",
        "bachelor %",
        "median_income",
        "white %",
        "black %",
        "native %",
        "asian %",
        "hispanic %",
        "poverty",
    ]
    for key in keys:
        all_data[key] = (
                all_data[key]
                + ((all_data[key] - data_2016[key]) * 2) * 2.0 / 3
                + ((all_data[key] - data_2012[key])) * 1.0 / 3
        )
    # 2012 CVAP is centered in 2010
    # 2016 CVAP is centered in 2014
    # 2020 CVAP is centered in 2018
    # 2024 CVAP is centered in 2022
    # 2024 = 2020 + (2020 - 2016)
    # 2024 = 2020 + (2020 - 2012) / 2
    all_data["CVAP"] = (
            all_data["CVAP"]
            + ((all_data["CVAP"] - data_2016["CVAP"])) * 2.0 / 3
            + ((all_data["CVAP"] - data_2012["CVAP"]) / 2) * 1.0 / 3
    )
    all_data["CVAP"] = np.clip(all_data["CVAP"], 10, np.inf)
    print("2024", all_data["CVAP"].sum())
    return all_data


def data_by_year():
    return {
        year: all_data(year)
        for year in (2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024)
    }


def ec():
    return pd.read_csv(os.path.join(CSVS, "ec.csv")).set_index("state")
