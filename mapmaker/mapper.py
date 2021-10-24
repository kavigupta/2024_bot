from abc import ABC, abstractmethod, abstractproperty

from cached_property import cached_property

import numpy as np
import plotly.graph_objects as go

from .data import counties, data_by_year
from .aggregation import get_state_results
from .colors import (
    BACKGROUND,
    COUNTY_SCALE_MARGIN_MIN,
    COUNTY_SCALE_MARGIN_MAX,
)
from .constants import TILT_MARGIN, LEAN_MARGIN, LIKELY_MARGIN
from .utils import counties_to_states


class BaseMap(ABC):
    @abstractproperty
    def counties(self):
        pass

    @abstractproperty
    def data(self):
        pass

    @cached_property
    def states(self):
        return counties_to_states(self.data, self.counties)

    def county_map(
        self, identifiers, *, variable_to_plot, zmid, zmin, zmax, colorscale
    ):
        figure = go.Choropleth(
            geojson=self.counties,
            locations=identifiers,
            z=variable_to_plot,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            marker_line_width=0,
            showscale=False,
        )
        return fit(figure)

    def map_county_margins(self, identifiers, *, dem_margin, profile):
        return self.county_map(
            identifiers,
            variable_to_plot=dem_margin,
            zmid=0,
            zmin=COUNTY_SCALE_MARGIN_MIN,
            zmax=COUNTY_SCALE_MARGIN_MAX,
            colorscale=profile.county_colorscale,
        )

    def map_county_demographics(self, identifiers, *, demographic_values):
        return self.county_map(
            identifiers,
            variable_to_plot=demographic_values,
            zmid=0.45,
            zmin=0,
            zmax=0.9,
            colorscale="jet",
        )

    def state_map(self, data, *, dem_margin, turnout, profile):
        state_margins = get_state_results(data, dem_margin=dem_margin, turnout=turnout)
        classes = [classify(m) for m in np.array(state_margins)]

        figure = go.Choropleth(
            geojson=self.states,
            z=np.array(classes),
            locations=state_margins.index,
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


class USABaseMap(BaseMap):
    @property
    def counties(self):
        return counties()

    @property
    def data(self):
        return data_by_year()[2020]


def fit(*figure):
    figure = go.Figure(
        figure,
    )
    figure.update_geos(scope="usa")
    figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return figure


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
