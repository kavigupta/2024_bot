from .constants import TILT_MARGIN
from .senate import d_lock_2022, r_lock_2022


def get_state_results(data, *, dem_margin, turnout, group_by="state"):
    data = data.copy()
    data["total_votes_predicted"] = turnout * data["CVAP"]
    data["total_margin"] = data["total_votes_predicted"] * dem_margin
    grouped = data.groupby(group_by).sum(min_count=1)
    grouped["total_margin"] = grouped["total_margin"] / grouped["total_votes_predicted"]
    return grouped["total_margin"]


def get_popular_vote(data, *, dem_margin, turnout):
    total_votes_predicted = turnout * data.CVAP
    return (total_votes_predicted * dem_margin).sum() / total_votes_predicted.sum()


def number_votes(data, *, turnout):
    return (turnout * data.CVAP).sum()


def get_electoral_vote(
    data, *, dem_margin, turnout, basemap, only_nonclose=False, **kwargs
):
    if only_nonclose:
        m = TILT_MARGIN
    else:
        m = 0
    ec_results = basemap.electoral_votes.join(
        get_state_results(data, turnout=turnout, dem_margin=dem_margin, **kwargs),
        how="inner",
    )

    return (
        ec_results["electoral_college"][ec_results.total_margin > m].sum(),
        ec_results["electoral_college"][ec_results.total_margin < -m].sum(),
    )


def get_senate_vote(data, *, dem_margin, turnout):
    state_results = get_state_results(data, turnout=turnout, dem_margin=dem_margin)
    dem_states = (state_results > 0).sum()
    gop_states = (state_results <= 0).sum()
    return dem_states + d_lock_2022, gop_states + r_lock_2022


def calculate_tipping_point(data, *, dem_margin, turnout, basemap, **kwargs):
    state_results = get_state_results(
        data, dem_margin=dem_margin, turnout=turnout, **kwargs
    )
    ec_results = basemap.electoral_votes.join(state_results, how="inner")
    dem_ec = ec_results["electoral_college"][ec_results.total_margin > 0].sum()
    gop_ec = ec_results["electoral_college"][ec_results.total_margin < 0].sum()

    total_ec = int(basemap.electoral_votes.sum())

    # Give the tiebreak to the GOP because of likely House delegation lean
    dem_needs = total_ec // 2 + (total_ec + 1) % 2
    gop_needs = total_ec // 2

    assert gop_needs in {total_ec - gop_needs, total_ec - gop_needs + 1}
    assert dem_needs in {total_ec - dem_needs + 1, total_ec - dem_needs + 2}

    tipping_point = None
    ec_total = 0
    if dem_ec >= dem_needs:
        # dem tipping pt
        for index, row in (
            ec_results[ec_results.total_margin > 0]
            .sort_values(by="total_margin", ascending=False)
            .iterrows()
        ):
            ec_total += row["electoral_college"]
            if ec_total >= dem_needs:
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
            if ec_total >= gop_needs:
                tipping_point = ec_results[
                    ec_results.index == index
                ].total_margin.reset_index()
                break
    if tipping_point is None:
        return "None", 0

    tipping_point_state, tipping_point_margin = (
        tipping_point.values[0][0],
        tipping_point.values[0][1],
    )

    return tipping_point_state, tipping_point_margin
