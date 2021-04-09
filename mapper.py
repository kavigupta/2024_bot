import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import us

from .data import counties

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


def state_map(data, dem_margin):
    data = data.copy()
    data["total_margin"] = data["total_votes"] * data[dem_margin]
    is_blue_state = data.groupby("state").sum()["total_margin"] > 0
    figure = go.Choropleth(
        locationmode="USA-states",
        z=np.array(is_blue_state).astype(np.float),
        locations=[us.states.lookup(x).abbr for x in is_blue_state.index],
        colorscale=[[0, "#f88"], [1, "#88f"]],
        #         marker_line_color="Blue" if is_blue else "Red",
        #         marker_line_width=2,
        showscale=False,
    )
    return fit(figure)
