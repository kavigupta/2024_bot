import plotly.graph_objects as go

import numpy as np
import us

from .data import counties
from .aggregation import get_state_results
from .colors import (
    BACKGROUND,
    COUNTY_SCALE_MARGIN_MIN,
    COUNTY_SCALE_MARGIN_MAX,
)
from .constants import TILT_MARGIN, LEAN_MARGIN, LIKELY_MARGIN


def fit(*figure):
    figure = go.Figure(
        figure,
    )
    figure.update_geos(scope="usa")
    figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return figure


def county_map(data, *, variable_to_plot, zmid, zmin, zmax, colorscale):
    figure = go.Choropleth(
        geojson=counties(),
        locations=data["FIPS"],
        z=variable_to_plot,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        marker_line_width=0,
        showscale=False,
    )
    return fit(figure)


def map_county_margins(data, *, dem_margin, profile):
    return county_map(
        data,
        variable_to_plot=dem_margin,
        zmid=0,
        zmin=COUNTY_SCALE_MARGIN_MIN,
        zmax=COUNTY_SCALE_MARGIN_MAX,
        colorscale=profile.county_colorscale,
    )


def map_county_demographics(data, *, demographic_values):
    return county_map(
        data,
        variable_to_plot=demographic_values,
        zmid=0.45,
        zmin=0,
        zmax=0.9,
        colorscale="jet",
    )


def classify(margin):
    if margin != margin:
        return np.nan
    if margin < -LIKELY_MARGIN:
        return 0
    if margin < -LEAN_MARGIN:
        return 0.15
    elif margin < -TILT_MARGIN:
        return 0.30
    elif margin < 0:
        return 0.45
    if margin < TILT_MARGIN:
        return 0.60
    elif margin < LEAN_MARGIN:
        return 0.75
    elif margin < LIKELY_MARGIN:
        return 0.90
    else:
        return 1


def state_map(data, *, dem_margin, turnout, profile):
    state_margins = get_state_results(data, dem_margin=dem_margin, turnout=turnout)
    classes = [classify(m) for m in np.array(state_margins)]

    figure = go.Choropleth(
        locationmode="USA-states",
        z=np.array(classes),
        locations=[us.states.lookup(x).abbr for x in state_margins.index],
        colorscale=[
            [0, profile.state_safe("gop")],
            [0.15, profile.state_likely("gop")],
            [0.30, profile.state_lean("gop")],
            [0.45, profile.state_tilt("gop")],
            [0.60, profile.state_tilt("dem")],
            [0.75, profile.state_lean("dem")],
            [0.90, profile.state_likely("dem")],
            [1, profile.state_safe("dem")],
        ],
        zmin=0,
        zmax=1,
        marker_line_width=2,
        showscale=False,
    )
    return fit(figure)
