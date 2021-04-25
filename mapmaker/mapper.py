import plotly.graph_objects as go

import numpy as np
import us

from .data import counties
from .aggregation import get_state_results
from .colors import (
    BACKGROUND,
    COUNTY_COLORSCALE,
    STATE_DEM,
    STATE_GOP,
    STATE_DEM_CLOSE,
    STATE_GOP_CLOSE,
)
from .constants import CLOSE_MARGIN


def fit(figure):
    figure = go.Figure(
        [figure],
    )
    figure.update_geos(scope="usa")
    figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return figure


def county_map(data, *, dem_margin):
    figure = go.Choropleth(
        geojson=counties(),
        locations=data["FIPS"],
        z=dem_margin,
        zmid=0,
        zmin=-1,
        zmax=1,
        colorscale=COUNTY_COLORSCALE,
        marker_line_width=0,
        name="margin",
        showscale=False,
    )
    return fit(figure)


def classify(margin):
    if margin < -CLOSE_MARGIN:
        return 0
    if margin > CLOSE_MARGIN:
        return 1
    if margin < 0:
        return 0.25
    else:
        return 0.75


def state_map(data, *, dem_margin, turnout):
    state_margins = get_state_results(data, dem_margin=dem_margin, turnout=turnout)
    classes = [classify(m) for m in np.array(state_margins)]

    figure = go.Choropleth(
        locationmode="USA-states",
        z=np.array(classes),
        locations=[us.states.lookup(x).abbr for x in state_margins.index],
        colorscale=[
            [0, STATE_GOP],
            [0.25, STATE_GOP_CLOSE],
            [0.75, STATE_DEM_CLOSE],
            [1, STATE_DEM],
        ],
        zmin=0,
        zmax=1,
        marker_line_width=2,
        showscale=False,
    )
    return fit(figure)
