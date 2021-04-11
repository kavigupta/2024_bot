import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import us

from .data import counties
from .processing import get_state_results

BACKGROUND = "#222"


def fit(figure):
    figure = go.Figure(
        [figure],
    )
    figure.update_geos(scope="usa")
    figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return figure


def county_map(data, dem_margin):
    figure = go.Choropleth(
        geojson=counties(),
        locations=data["FIPS"],
        z=data[dem_margin],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorscale=[
            [0, "#f00"],
            [0.499, "#fcc"],
            [0.5, "white"],
            [0.501, "#ccf"],
            [1.0, "#00f"],
        ],
        marker_line_width=0,
        name="margin",
        showscale=False,
    )
    return fit(figure)


def classify(margin):
    if margin < -0.5e-2:
        return -10
    if margin > 0.5e-2:
        return 10
    if margin < 0:
        return -7
    else:
        return 7


def state_map(data, dem_margin):
    state_margins = get_state_results(data, dem_margin)
    classes = [classify(m) for m in np.array(state_margins)]

    figure = go.Choropleth(
        locationmode="USA-states",
        z=np.array(classes),
        locations=[us.states.lookup(x).abbr for x in state_margins.index],
        colorscale=[[0, "#f88"], [0.5, "white"], [1, "#f88"]],
        zmin=-10,
        zmax=10,
        marker_line_width=2,
        showscale=False,
    )
    return fit(figure)
