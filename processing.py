from .data import ec


def get_state_results(data, dem_margin):
    data = data.copy()
    data["total_margin"] = data["total_votes"] * data[dem_margin]
    return data.groupby("state").sum()["total_margin"] > 0


def get_popular_vote(data, dem_margin):
    return (data["total_votes"] * data[dem_margin]).sum() / data["total_votes"].sum()


def get_electoral_vote(data, dem_margin):
    ec_results = ec().join(get_state_results(data, dem_margin), how="inner")
    return (
        ec_results["electoral_college"][ec_results.total_margin].sum(),
        ec_results["electoral_college"][~ec_results.total_margin].sum(),
    )
