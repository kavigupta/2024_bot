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


@lru_cache(None)
def all_data(demographic_projection=False):
    swing_2012_2016 = pd.read_csv(f"{CSVS}/2012 to 2016 swing.csv")
    demo_2012 = pd.read_csv(f"{CSVS}/2012_demographics_votes.csv")
    demo_2016 = pd.read_csv(f"{CSVS}/2016_demographics_votes.csv")
    demo_2020 = pd.read_csv(
        f"{CSVS}/2020_demographics_votes_fips.csv",
        dtype=dict(FIPS=str),
    )
    demo_2020["FIPS"] = demo_2020["FIPS"].map(lambda x: x if len(x) == 5 else "0" + x)
    relevant_demo_2012 = demo_2012[
        [
            "gisjoin",
            "swing2008to12",
            "obamamargin2012",
            "medianage_2012",
            "bachelorabove_2012",
            "medianincome_2012",
            "white_2012",
            "black_2012",
            "native_2012",
            "asian_2012",
            "hispanic_2012",
            "otherrace_2012",
            "evangelical_2012",
            "mainlineprotestant_2012",
            "catholic_2012",
            "mormon_2012",
            "other religion",
            "2012votes",
            "County Power Index 2012",
        ]
    ]

    relevant_demo_2016 = demo_2016[
        ["gisjoin", "Total Population 2016", "2016_votes", "County Power Index 2016"]
    ]

    relevant_swing_2016 = swing_2012_2016[
        [
            "gisjoin",
            "Obama 2012 %",
            "Romney 2012 %",
            "Other 2012 %",
            "Obama 2-Party Margin",
            "Clinton 2016 %",
            "Trump 2016 %",
            "Other 2016 %",
            "Clinton 2-Party Margin",
            "2012 to 2016 2-Party Swing",
        ]
    ]

    relevant_demo_2020 = demo_2020[
        [
            "FIPS",
            "gisjoin",
            "state",
            "Median Age 2018",
            "% Bachelor Degree or Above",
            "Median Household Income",
            "Total Population",
            "White %",
            "Black %",
            "Native %",
            "Asian %",
            "Multiracial %",
            "Pacific Islander %",
            "Hispanic %",
            "Rural % (2010)",
            "Total Adherents (All Types) Per 1000 Population (2010)",
            "Evangelical Per 1000 (2010)",
            "Black Protestant Per 1000 (2010)",
            "Mainline Protestant Per 1000 (2010)",
            "Catholic Per 1000 (2010)",
            "Orthodox Christian Per 1000 (2010)",
            "Other Religion (Non-Christian) Per 1000 (2010)",
            "Buddhist Per 1000 (2010)",
            "Mormon Per 1000 (2010)",
            "Hindu Per 1000 (2010)",
            "Muslim Per 1000 (2010)",
            "Orthodox Jewish Per 1000 (2010)",
            "Reform/Reconstructionist Jewish Per 1000 (2010)",
            "2018 votes",
            "2018 partisanship",
        ]
    ]

    relevant_demo_2020.insert(
        1, "total_votes", demo_2020["Total Votes 2020 (AK is Rough Estimate)"]
    )

    relevant_demo_2020.insert(1, "biden_2020", demo_2020["Biden 2020 Margin"])
    all_data = (
        relevant_demo_2020.merge(relevant_demo_2012, how="inner")
        .merge(relevant_swing_2016, how="inner")
        .merge(relevant_demo_2016, how="inner")
    )

    del all_data["gisjoin"]

    ## PROJECTIONS (2018 --> 2024)
    if demographic_projection:
        all_data["Total Population"] = (
            all_data["Total Population"]
            + (all_data["Total Population"] - all_data["Total Population 2016"]) * 2.5
        )
        all_data["White %"] = (
            all_data["White %"] + (all_data["White %"] - all_data["white_2012"])
        ).clip(0, 1)
        all_data["Black %"] = all_data["Black %"] + (
            all_data["Black %"] - all_data["black_2012"]
        ).clip(0, 1)
        all_data["Hispanic %"] = all_data["Hispanic %"] + (
            all_data["Hispanic %"] - all_data["hispanic_2012"]
        ).clip(0, 1)
        all_data["Asian %"] = all_data["Asian %"] + (
            all_data["Asian %"] - all_data["asian_2012"]
        ).clip(0, 1)
        all_data["% Bachelor Degree or Above"] = all_data[
            "% Bachelor Degree or Above"
        ] + (
            all_data["% Bachelor Degree or Above"] - all_data["bachelorabove_2012"]
        ).clip(
            0, 1
        )
        all_data["Median Household Income"] = all_data["Median Household Income"] + (
            all_data["Median Household Income"] - all_data["medianincome_2012"]
        )

    ## Nonlinearity
    all_data["county_diversity_black_white"] = all_data["Black %"] * all_data["White %"]
    all_data["county_diversity_hispanic_white"] = (
        all_data["Hispanic %"] * all_data["White %"]
    )
    all_data["county_diversity_white_homogenity"] = all_data["White %"] ** 2
    all_data["county_diversity_hispanic_homogenity"] = all_data["Hispanic %"] ** 2
    all_data["county_diversity_native_homogenity"] = all_data["Native %"] ** 2

    def logify(column):
        all_data[column] = np.log(all_data[column]).replace(-np.inf, -1000)

    logify("Median Household Income")
    logify("Total Population")
    logify("Total Population 2016")
    logify("2012votes")
    logify("2016_votes")
    logify("2018 votes")
    all_data["turnout_spike"] = all_data["2018 votes"] / all_data["2016_votes"]
    all_data['hispanic_rural'] = all_data['Hispanic %'] ** 2 * all_data['Rural % (2010)']

    return all_data


def data_by_year():
    return {
        2020: all_data(demographic_projection=False),
        2024: all_data(demographic_projection=True),
    }


def ec():
    return pd.read_csv(os.path.join(CSVS, "ec.csv")).set_index("state")
