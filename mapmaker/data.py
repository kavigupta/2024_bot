import os
from functools import lru_cache

from urllib.request import urlopen
import json

import numpy as np
import pandas as pd

NAT_REG_MODEL = os.environ["NAT_REG_MODEL"]

current_folder = os.path.dirname(__file__)


@lru_cache(None)
def counties():
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        return json.load(response)


@lru_cache(None)
def all_data():
    swing_2012_2016 = pd.read_csv(f"{NAT_REG_MODEL}/2012 to 2016 swing.csv")
    demo_2012 = pd.read_csv(f"{NAT_REG_MODEL}/2012_demographics_votes.csv")
    demo_2020 = pd.read_csv(
        f"{NAT_REG_MODEL}/2020_demographics_votes_fips.csv",
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
            "rural",
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
        ]
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
            "Rural % (2010)",
            "Median Age 2018",
            "% Bachelor Degree or Above 2018",
            "Median Household Income 2018",
            "Total Population 2018",
            "White CVAP % 2018",
            "Black CVAP % 2018",
            "Native CVAP % 2018",
            "Asian CVAP % 2018",
            "Multiracial CVAP % 2018",
            "Pacific Islander CVAP % 2018",
            "Hispanic CVAP % 2018",
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
        ]
    ]
    relevant_demo_2020.insert(
        1, "total_votes", demo_2020["Total Votes 2020 (AK is Rough Estimate)"]
    )
    relevant_demo_2020.insert(1, "biden_2020", demo_2020["Biden 2020 Margin"])
    all_data = relevant_demo_2020.merge(relevant_demo_2012, how="inner").merge(
        relevant_swing_2016, how="inner"
    )
    del all_data["gisjoin"]

    ## Demographic Change
    all_data["black_pop_change"] = (
        all_data["White CVAP % 2018"] - all_data["white_2012"]
    ) * all_data["Total Population 2018"]
    all_data["white_pop_change"] = (
        all_data["Black CVAP % 2018"] - all_data["black_2012"]
    ) * all_data["Total Population 2018"]
    all_data["hispanic_pop_change"] = (
        all_data["Hispanic CVAP % 2018"] - all_data["hispanic_2012"]
    ) * all_data["Total Population 2018"]
    all_data["changefrom_bachelorabove_2012"] = (
        all_data["% Bachelor Degree or Above 2018"] - all_data["bachelorabove_2012"]
    ) / all_data["bachelorabove_2012"]

    ## Nonlinearity
    all_data["county_diversity_black_white"] = (
        all_data["Black CVAP % 2018"] * all_data["White CVAP % 2018"]
    )
    all_data["county_diversity_hispanic_white"] = (
        all_data["Hispanic CVAP % 2018"] * all_data["White CVAP % 2018"]
    )
    all_data["Median Household Income 2018"] = np.log(
        all_data["Median Household Income 2018"]
    ).replace(-np.inf, -1000)
    all_data["Total Population 2018"] = np.log(
        all_data["Total Population 2018"]
    ).replace(-np.inf, -1000)

    return all_data


def ec():
    return pd.read_csv(os.path.join(current_folder, "ec.csv")).set_index("state")
