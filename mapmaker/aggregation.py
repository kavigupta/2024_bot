from .data import ec
from .constants import TILT_MARGIN


def get_state_results(data, *, dem_margin, turnout):
    data = data.copy()
    data["total_votes_predicted"] = turnout * data["CVAP"]
    data["total_margin"] = data["total_votes_predicted"] * dem_margin
    grouped = data.groupby("state").sum()
    grouped["total_margin"] = grouped["total_margin"] / grouped["total_votes_predicted"]
    return grouped["total_margin"]


def get_popular_vote(data, *, dem_margin, turnout):
    total_votes_predicted = turnout * data.CVAP
    return (total_votes_predicted * dem_margin).sum() / total_votes_predicted.sum()


def number_votes(data, *, turnout):
    return (turnout * data.CVAP).sum()


def get_electoral_vote(data, *, dem_margin, turnout, only_nonclose=False):
    if only_nonclose:
        m = TILT_MARGIN
    else:
        m = 0
    ec_results = ec().join(
        get_state_results(data, turnout=turnout, dem_margin=dem_margin), how="inner"
    )

    return (
        ec_results["electoral_college"][ec_results.total_margin > m].sum(),
        ec_results["electoral_college"][ec_results.total_margin < -m].sum(),
    )


def calculate_tipping_point(data, *, dem_margin, turnout):
    state_results = get_state_results(data, dem_margin=dem_margin, turnout=turnout)
    ec_results = ec().join(state_results, how="inner")
    dem_ec = ec_results["electoral_college"][ec_results.total_margin > 0].sum()
    gop_ec = ec_results["electoral_college"][ec_results.total_margin < 0].sum()
    tipping_point = None
    ec_total = 0
    if dem_ec >= 270:
        # dem tipping pt
        for index, row in (
            ec_results[ec_results.total_margin > 0]
            .sort_values(by="total_margin", ascending=False)
            .iterrows()
        ):
            ec_total += row["electoral_college"]
            if ec_total >= 270:
                tipping_point = ec_results[
                    ec_results.index == index
                ].total_margin.reset_index()
                break
    else:
        # GOP tipping pt
        for index, row in (
            ec_results[ec_results.total_margin < 0]
            .sort_values(by="total_margin", ascending=True)
            .iterrows()
        ):
            ec_total += row["electoral_college"]
            if ec_total >= 269:
                # Give the tiebreak to the GOP because of likely House delegation lean
                tipping_point = ec_results[
                    ec_results.index == index
                ].total_margin.reset_index()
                break

    tipping_point_state, tipping_point_margin = (
        tipping_point.values[0][0],
        tipping_point.values[0][1],
    )

    return tipping_point_state, tipping_point_margin
