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
    get_electoral_vote_by_voteshare,
    get_senate_vote,
    get_state_results_by_voteshare,
)
from .colors import BACKGROUND
from .constants import TILT_MARGIN, LEAN_MARGIN, LIKELY_MARGIN
from .utils import counties_to_states
from .text import draw_text
from .senate import senate_2022
from .data import ec


class BaseMap(ABC):
    @abstractproperty
    def counties(self):
        pass

    @abstractproperty
    def metadata(self):
        pass

    @abstractproperty
    def county_plotly_kwargs(self):
        pass

    @abstractproperty
    def state_plotly_kwargs(self):
        pass

    @abstractmethod
    def draw_topline(
        self, voteshare_by_party, turnout, *, draw, scale, profile, text_center, y
    ):
        pass

    @abstractmethod
    def draw_tipping_point(
        self, data, voteshare_by_party, turnout, *, draw, scale, profile, text_center, y
    ):
        pass

    @abstractmethod
    def county_mask(self, year):
        pass

    @abstractproperty
    def electoral_votes(self):
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

    def map_county_margins(self, identifiers, *, voteshare_by_party, profile):
        return self.county_map(
            identifiers,
            variable_to_plot=profile.place_on_county_colorscale(voteshare_by_party),
            zmin=0,
            zmid=0.5,
            zmax=1,
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

    def state_map(self, data, *, voteshare_by_party, turnout, profile):
        index, state_voteshares = get_state_results_by_voteshare(
            data, voteshare_by_party=voteshare_by_party, turnout=turnout
        )
        classes = [
            profile.place_on_state_colorscale({k: [v[k]] for k in v})[0]
            for v in state_voteshares
        ]

        figure = go.Choropleth(
            geojson=self.states,
            z=np.array(classes),
            locations=index,
            colorscale=profile.state_colorscale,
            zmin=0,
            zmax=1,
            showscale=False,
            **self.state_plotly_kwargs,
        )
        return fit(figure, modify_figure_layout=self.modify_figure_layout)

    def populate(self, data, voteshare_by_party, turnout):
        return PopulatedMap(self, data, voteshare_by_party, turnout)


@attr.s
class PopulatedMap:
    basemap = attr.ib()
    data = attr.ib()
    voteshare_by_party = attr.ib()
    turnout = attr.ib()

    def draw_topline(self, **kwargs):
        return self.basemap.draw_topline(
            self.data, self.voteshare_by_party, self.turnout, **kwargs
        )

    def draw_tipping_point(self, **kwargs):
        return self.basemap.draw_tipping_point(
            self.data, self.voteshare_by_party, self.turnout, **kwargs
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

    @property
    def state_plotly_kwargs(self):
        return dict(marker_line_width=2)

    def county_mask(self, year):
        return 1

    @property
    def electoral_votes(self):
        return ec()

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
        return draw_ec(self, *args, **kwargs)

    def draw_tipping_point(self, *args, **kwargs):
        return draw_tipping_point(self, *args, **kwargs)


class USASenateBaseMap(USABaseMap):
    def county_mask(self, year):
        assert year == 2022
        return self.metadata.state.apply(
            lambda x: 1 if us.states.lookup(x).abbr in senate_2022 else np.nan
        )

    def draw_topline(
        self, data, voteshare_by_party, turnout, *, draw, scale, profile, text_center, y
    ):
        dem_senate, gop_senate = get_senate_vote(
            data, voteshare_by_party=voteshare_by_party, turnout=turnout
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
        self, data, voteshare_by_party, turnout, *, draw, scale, profile, text_center, y
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
    if margin < TILT_MARGIN:
        return 0.1
    elif margin < LEAN_MARGIN:
        return 0.2
    elif margin < LIKELY_MARGIN:
        return 0.3
    else:
        return 0.4


def draw_ec(
    basemap,
    data,
    voteshare_by_party,
    turnout,
    *,
    draw,
    scale,
    profile,
    text_center,
    y,
    **kwargs,
):
    ec_by_party = get_electoral_vote_by_voteshare(
        data, voteshare_by_party=voteshare_by_party, turnout=turnout, basemap=basemap
    )
    safe_ec_by_party = get_electoral_vote_by_voteshare(
        data,
        voteshare_by_party=voteshare_by_party,
        turnout=turnout,
        basemap=basemap,
        only_nonclose=True,
    )
    close_ec_by_party = {
        party: ec_by_party[party] - safe_ec_by_party[party] for party in ec_by_party
    }
    for k in close_ec_by_party:
        assert close_ec_by_party[k] >= 0

    # TODO support more than 2 parties
    draw_text(
        draw,
        40 * scale,
        [
            (str(ec_by_party["dem"]), profile.state_safe("dem")),
            (" - ", profile.text_color),
            (str(ec_by_party["gop"]), profile.state_safe("gop")),
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
            (str(close_ec_by_party["dem"]), profile.state_tilt("dem")),
            (" - ", profile.text_color),
            (str(close_ec_by_party["gop"]), profile.state_tilt("gop")),
        ],
        text_center * scale,
        y * scale,
        align=("center"),
    )
    return y


def draw_tipping_point(
    basemap,
    data,
    voteshare_by_party,
    turnout,
    *,
    draw,
    scale,
    profile,
    text_center,
    y,
):
    (
        tipping_point_state,
        tipping_point_party,
        tipping_point_margin,
    ) = calculate_tipping_point(
        data,
        voteshare_by_party=voteshare_by_party,
        turnout=turnout,
        basemap=basemap,
        extras=profile.extra_ec,
    )

    if tipping_point_party is None:
        tipping_point_str = "None"
        tipping_point_color = profile.text_color
    else:
        tipping_point_str = f"{tipping_point_state} {profile.symbol[tipping_point_party]}+{tipping_point_margin:.2%}"
        tipping_point_color = profile.state_tilt(tipping_point_party)

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
