from abc import ABC, abstractmethod, abstractproperty
import os

import attr
import us

import svgutils.transform as sg
from cached_property import cached_property

import numpy as np
import plotly.graph_objects as go


from .data import counties, data_by_year
from .aggregation import (
    calculate_tipping_point,
    get_electoral_vote,
    get_senate_vote,
    get_state_results,
)
from .colors import (
    BACKGROUND,
    COUNTY_SCALE_MARGIN_MIN,
    COUNTY_SCALE_MARGIN_MAX,
)
from .constants import TILT_MARGIN, LEAN_MARGIN, LIKELY_MARGIN
from .utils import counties_to_states
from .text import draw_text
from .senate import senate_2022


class BaseMap(ABC):
    @abstractproperty
    def counties(self):
        pass

    @abstractproperty
    def metadata(self):
        pass

    @abstractmethod
    def county_plotly_kwargs(self, figure):
        pass

    @abstractmethod
    def draw_topline(
        self, dem_margin, turnout, *, draw, scale, profile, text_center, y
    ):
        pass

    @abstractmethod
    def draw_tipping_point(
        self, data, dem_margin, turnout, *, draw, scale, profile, text_center, y
    ):
        pass

    @abstractmethod
    def county_mask(self, year):
        pass

    @property
    def extra_county_maps(self):
        return []

    @property
    def map_scale(self):
        return 1

    @property
    def map_dy(self):
        return 0

    @cached_property
    def states(self):
        return counties_to_states(self.metadata, self.counties)

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
            **self.county_plotly_kwargs,
            showscale=False,
        )
        return fit(figure, modify_figure_layout=self.modify_figure_layout)

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
        return fit(figure, modify_figure_layout=self.modify_figure_layout)

    def populate(self, data, dem_margin, turnout):
        return PopulatedMap(self, data, dem_margin, turnout)


@attr.s
class PopulatedMap:
    basemap = attr.ib()
    data = attr.ib()
    dem_margin = attr.ib()
    turnout = attr.ib()

    def draw_topline(self, **kwargs):
        return self.basemap.draw_topline(
            self.data, self.dem_margin, self.turnout, **kwargs
        )

    def draw_tipping_point(self, **kwargs):
        return self.basemap.draw_tipping_point(
            self.data, self.dem_margin, self.turnout, **kwargs
        )


class USABaseMap(BaseMap):
    @property
    def counties(self):
        return counties()

    @property
    def metadata(self):
        data = data_by_year()[2020]
        return data[["FIPS", "state"]]

    @property
    def county_plotly_kwargs(self):
        return dict(marker_line_width=0)

    def county_mask(self, year):
        return 1

    def modify_figure_layout(self, figure):
        figure.update_geos(scope="usa")
        figure.update_layout(geo=dict(bgcolor=BACKGROUND, lakecolor=BACKGROUND))

    @property
    def extra_county_maps(self):
        return [
            sg.fromfile(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../stateboundariesinternal.svg",
                )
            ).getroot()
        ]


class USAPresidencyBaseMap(USABaseMap):
    def draw_topline(self, *args, **kwargs):
        return draw_ec(*args, **kwargs)

    def draw_tipping_point(self, *args, **kwargs):
        return draw_tipping_point(*args, **kwargs)


class USASenateBaseMap(USABaseMap):
    def county_mask(self, year):
        assert year == 2022
        return self.metadata.state.apply(
            lambda x: 1 if us.states.lookup(x).abbr in senate_2022 else np.nan
        )

    def draw_topline(
        self, data, dem_margin, turnout, *, draw, scale, profile, text_center, y
    ):
        dem_senate, gop_senate = get_senate_vote(
            data, dem_margin=dem_margin, turnout=turnout
        )
        y += 15 // 2 + 20
        draw_text(
            draw,
            40 * scale,
            [
                (str(dem_senate), profile.state_safe("dem")),
                (" - ", profile.text_color),
                (str(gop_senate), profile.state_safe("gop")),
            ],
            text_center * scale,
            y * scale,
            align=("center", 1),
        )
        return y

    def draw_tipping_point(
        self, data, dem_margin, turnout, *, draw, scale, profile, text_center, y
    ):
        pass


def fit(*figure, modify_figure_layout):
    figure = go.Figure(
        figure,
    )
    modify_figure_layout(figure)
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


def draw_ec(
    data, dem_margin, turnout, *, draw, scale, profile, text_center, y, **kwargs
):
    dem_ec, gop_ec = get_electoral_vote(
        data, dem_margin=dem_margin, turnout=turnout, **kwargs
    )
    dem_ec_safe, gop_ec_safe = get_electoral_vote(
        data, dem_margin=dem_margin, turnout=turnout, only_nonclose=True, **kwargs
    )
    dem_ec_close, gop_ec_close = dem_ec - dem_ec_safe, gop_ec - gop_ec_safe
    assert dem_ec_close >= 0 and gop_ec_close >= 0
    draw_text(
        draw,
        40 * scale,
        [
            (str(dem_ec), profile.state_safe("dem")),
            (" - ", profile.text_color),
            (str(gop_ec), profile.state_safe("gop")),
        ],
        text_center * scale,
        y * scale,
        align=("center", 1),
    )

    y += 15 // 2 + 20

    draw_text(
        draw,
        15 * scale,
        [
            ("Close: ", profile.text_color),
            (str(dem_ec_close), profile.state_tilt("dem")),
            (" - ", profile.text_color),
            (str(gop_ec_close), profile.state_tilt("gop")),
        ],
        text_center * scale,
        y * scale,
        align=("center"),
    )
    return y


def draw_tipping_point(
    data, dem_margin, turnout, *, draw, scale, profile, text_center, y, **kwargs
):
    tipping_point_state, tipping_point_margin = calculate_tipping_point(
        data, dem_margin=dem_margin, turnout=turnout, **kwargs
    )
    tipping_point_str = None
    tipping_point_color = None

    if tipping_point_margin > 0:
        tipping_point_str = (
            f"{tipping_point_state} {profile.symbol['dem']}+{tipping_point_margin:.2%}"
        )
        tipping_point_color = profile.state_tilt("dem")
    else:
        tipping_point_str = (
            f"{tipping_point_state} {profile.symbol['gop']}+{-tipping_point_margin:.2%}"
        )
        tipping_point_color = profile.state_tilt("gop")

    draw_text(
        draw,
        10 * scale,
        [
            ("Tipping Point: ", profile.text_color),
            (tipping_point_str, tipping_point_color),
        ],
        text_center * scale,
        y * scale,
        align=("center"),
    )
