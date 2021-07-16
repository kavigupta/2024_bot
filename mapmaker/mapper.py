import plotly.graph_objects as go

import numpy as np
import us

from .data import counties
from .aggregation import get_state_results
from .colors import (
    BACKGROUND,
    COUNTY_COLORSCALE,
    COUNTY_SCALE_MARGIN_MIN,
    COUNTY_SCALE_MARGIN_MAX,
    STATE_GOP,
    STATE_DEM,
    STATE_GOP_BATTLEGROUND,
    STATE_DEM_BATTLEGROUND,
    STATE_GOP_CLOSE,
    STATE_DEM_CLOSE,
)
from .constants import CLOSE_MARGIN, BATTLEGROUND_MARGIN, BLOWOUT_MARGIN


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


def map_county_margins(data, *, dem_margin):
    return county_map(
        data,
        variable_to_plot=dem_margin,
        zmid=0,
        zmin=COUNTY_SCALE_MARGIN_MIN,
        zmax=COUNTY_SCALE_MARGIN_MAX,
        colorscale=COUNTY_COLORSCALE,
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
    if margin < -BATTLEGROUND_MARGIN:
        return 0
    if margin < -CLOSE_MARGIN:
        return 0.20
    elif margin < 0:
        return 0.40
    if margin < CLOSE_MARGIN:
        return 0.60
    elif margin < BATTLEGROUND_MARGIN:
        return 0.80
    else:
        return 1


def state_map(data, *, dem_margin, turnout):
    state_margins = get_state_results(data, dem_margin=dem_margin, turnout=turnout)
    classes = [classify(m) for m in np.array(state_margins)]

    figure = go.Choropleth(
        locationmode="USA-states",
        z=np.array(classes),
        locations=[us.states.lookup(x).abbr for x in state_margins.index],
        colorscale=[
            [0, STATE_GOP],
            [0.20, STATE_GOP_BATTLEGROUND],
            [0.40, STATE_GOP_CLOSE],
            [0.60, STATE_DEM_CLOSE],
            [0.80, STATE_DEM_BATTLEGROUND],
            [1, STATE_DEM],
        ],
        zmin=0,
        zmax=1,
        marker_line_width=2,
        showscale=False,
    )
    return fit(figure)
