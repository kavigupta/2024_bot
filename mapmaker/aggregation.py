import numpy as np
from mapmaker.utils import dict_argmax
from .constants import TILT_MARGIN
from .senate import d_lock_2022, r_lock_2022


def get_state_results_by_voteshare(data, *, voteshare_by_party, turnout, **kwargs):
    result = {
        party: get_state_results(
            data, dem_margin=voteshare_by_party[party], turnout=turnout, **kwargs
        )
        for party in voteshare_by_party
    }
    indices = [x.index for x in result.values()]
    for idxs in indices:
        assert (idxs == indices[0]).all()
    idxs = indices[0]
    return idxs, [{k: result[k][idx] for k in result} for idx in idxs]


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


def get_popular_vote_by_voteshare(data, *, voteshare_by_party, turnout):
    return {
        k: get_popular_vote(data, dem_margin=voteshare_by_party[k], turnout=turnout)
        for k in voteshare_by_party
    }


def number_votes(data, *, turnout):
    return (turnout * data.CVAP).sum()


def get_electoral_vote(data, *, dem_margin, turnout, basemap, only_nonclose=False):
    if only_nonclose:
        m = TILT_MARGIN
    else:
        m = 0
    ec_results = basemap.electoral_votes.join(
        get_state_results(
            data,
            turnout=turnout,
            dem_margin=dem_margin,
            group_by=basemap.electoral_votes.index.name,
        ),
        how="inner",
    )

    return (
        ec_results["electoral_college"][ec_results.total_margin > m].sum(),
        ec_results["electoral_college"][ec_results.total_margin < -m].sum(),
    )


def get_electoral_vote_by_voteshare(
    data, *, voteshare_by_party, turnout, basemap, only_nonclose=False
):
    if only_nonclose:
        m = TILT_MARGIN
    else:
        m = 0
    ec_results_by_party = {
        party: basemap.electoral_votes.join(
            get_state_results(
                data,
                turnout=turnout,
                dem_margin=voteshare_by_party[party],
                group_by=basemap.electoral_votes.index.name,
            ),
            how="inner",
        )
        for party in voteshare_by_party
    }

    ec_by_state = dict(
        zip(basemap.electoral_votes.index, basemap.electoral_votes.electoral_college)
    )

    total_ec = {}

    for party in ec_results_by_party:
        total_ec[party] = 0
        for state in ec_results_by_party[party].index:
            other_results = [
                ec_results_by_party[p].loc[state].total_margin
                for p in ec_results_by_party
                if p != party
            ]
            if (
                ec_results_by_party[party].loc[state].total_margin
                - np.max(other_results)
                > m
            ):
                total_ec[party] += ec_by_state[state]

    return total_ec


def get_senate_vote(data, *, voteshare_by_party, turnout):
    _, state_results = get_state_results_by_voteshare(
        data, turnout=turnout, voteshare_by_party=voteshare_by_party
    )
    assert voteshare_by_party.keys() == {"dem", "gop"}
    argmaxes = [dict_argmax(x) for x in state_results]
    dem_states = sum(x == "dem" for x in argmaxes)
    gop_states = sum(x == "gop" for x in argmaxes)
    return dem_states + d_lock_2022, gop_states + r_lock_2022


def calculate_tipping_point(
    data, *, voteshare_by_party, turnout, basemap, extras, **kwargs
):
    state_names, state_results = get_state_results_by_voteshare(
        data,
        voteshare_by_party=voteshare_by_party,
        turnout=turnout,
        group_by=basemap.electoral_votes.index.name,
    )
    ec_by_state = dict(
        zip(basemap.electoral_votes.index, basemap.electoral_votes.electoral_college)
    )
    total_ec = sum(ec_by_state.values())
    needed_ec = total_ec // 2 + 1

    winning_margin = [
        to_winning_margin({k: [v[k]] for k in v})[0] for v in state_results
    ]
    winning_margin_with_state = [
        (party, margin, state)
        for (party, margin), state in zip(winning_margin, state_names)
    ]

    ec_by_party = {
        party: sum(
            ec_by_state[state]
            for p, _, state in winning_margin_with_state
            if p == party
        )
        for party in voteshare_by_party
    }
    parties = sorted(ec_by_party)
    winning_party = parties[np.argmax([ec_by_party[p] for p in parties])]

    if ec_by_party[winning_party] < needed_ec:
        return None, None, None

    states_won_by_winning_party = [
        (party, margin, state)
        for party, margin, state in winning_margin_with_state
        if party == winning_party
    ]

    ec = extras.get(winning_party, 0)
    for party, margin, state in sorted(
        states_won_by_winning_party, key=lambda x: -x[1]
    ):
        ec += ec_by_state[state]
        if ec >= needed_ec:
            return state, party, margin
    assert False, "not reachable"


def to_winning_margin(voteshare_by_party):
    parties = sorted(voteshare_by_party)
    out = []
    for results in zip(*[voteshare_by_party[k] for k in parties]):
        assert len(results) == len(parties)
        idx = np.argmax(results)
        out.append(
            (parties[idx], results[idx] - np.max([*results[:idx], *results[idx + 1 :]]))
        )
    return out

def to_winning_margin_single(voteshare_by_party):
    [result] = to_winning_margin(
            {k: [v] for k, v in voteshare_by_party.items()}
        )
    return result