from .data import ec
from .constants import CLOSE_MARGIN


def get_state_results(data, dem_margin):
    data = data.copy()
    data["total_margin"] = data["total_votes"] * dem_margin
    grouped = data.groupby("state").sum()
    grouped["total_margin"] = grouped["total_margin"] / grouped["total_votes"]
    return grouped["total_margin"]


def get_popular_vote(data, dem_margin):
    return (data["total_votes"] * dem_margin).sum() / data["total_votes"].sum()


def get_electoral_vote(data, dem_margin, only_nonclose=False):
    if only_nonclose:
        m = CLOSE_MARGIN
    else:
        m = 0
    ec_results = ec().join(get_state_results(data, dem_margin), how="inner")
    return (
        ec_results["electoral_college"][ec_results.total_margin > m].sum(),
        ec_results["electoral_college"][ec_results.total_margin < -m].sum(),
    )
