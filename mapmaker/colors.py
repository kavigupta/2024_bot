import colorsys

import numpy as np
import matplotlib.colors
from PIL import ImageColor

import attr

import plotly.colors

BACKGROUND = "#222"
BACKGROUND_RGB = "rgb(34, 34, 34)"

COUNTY_SCALE_MARGIN_MAX = 0.8

MARGINAL = 0.001
INTERVAL = 0.1


DEFAULT_CREDIT = "by @notkavi and @lxeagle17 with data from @mill226"


@attr.s
class Profile:
    symbol = attr.ib()
    hue = attr.ib()
    bot_name = attr.ib()
    credit = attr.ib()
    name = attr.ib(default=None)
    text_color = attr.ib(default=None)
    extra_ec = attr.ib(default={})

    def color(self, party, saturation):
        rgb = np.array(colorsys.hsv_to_rgb(self.hue[party], saturation, 1))
        return "#%02x%02x%02x" % tuple((rgb * 255).astype(np.uint8))

    def county_max(self, party):
        return self.color(party, 1)

    def county_min(self, party):
        return self.color(party, 1 / 3)

    def state_tilt(self, party):
        return self.color(party, 0.18823529411764706)

    def state_lean(self, party):
        return self.color(party, 0.38431372549019605)

    def state_likely(self, party):
        return self.color(party, 0.584313725490196)

    def state_safe(self, party):
        return self.color(party, 0.7803921568627451)

    def place_on_county_colorscale(self, voteshare_by_party):
        from .aggregation import to_winning_margin

        winning_margin = to_winning_margin(voteshare_by_party)
        parties = sorted(self.symbol)
        out = []
        for party, margin in winning_margin:
            out.append(
                (
                    2 * parties.index(party)
                    + 1
                    - min(margin / COUNTY_SCALE_MARGIN_MAX, 1)
                )
                * INTERVAL
            )
        return np.array(out)

    def place_on_state_colorscale(self, voteshare_by_party):
        from .aggregation import to_winning_margin
        from mapmaker.mapper import classify

        parties = sorted(self.symbol)
        out = []
        for party, margin in to_winning_margin(voteshare_by_party):
            out.append(INTERVAL * (parties.index(party) + classify(margin)))

        return np.array(out)

    @property
    def county_colorscale(self):
        parties = sorted(self.symbol)
        assert (2 * len(parties) + 1) * INTERVAL < 1
        result = []
        for idx, party in enumerate(parties):
            result += [
                [2 * idx * INTERVAL, self.county_max(party)],
                [
                    (2 * idx + 1) * INTERVAL - MARGINAL * INTERVAL * 2,
                    self.county_min(party),
                ],
                [(2 * idx + 1) * INTERVAL, "#ffffff"],
            ]

        result = result + [[1.0, "#ffffff"]]
        return result

    @property
    def state_colorscale(self):
        parties = sorted(self.symbol)
        out = [(0, "#000000")]
        for idx, party in enumerate(parties):
            out += [
                [INTERVAL * (idx + 0.1), self.state_tilt(party)],
                [INTERVAL * (idx + 0.2), self.state_lean(party)],
                [INTERVAL * (idx + 0.3), self.state_likely(party)],
                [INTERVAL * (idx + 0.4), self.state_safe(party)],
            ]
        out += [[1, "#ffffff"]]
        return out


STANDARD_PROFILE = Profile(
    symbol=dict(dem="D", gop="R"),
    hue=dict(dem=2 / 3, gop=1),
    bot_name="bot_2024",
    credit=DEFAULT_CREDIT,
    extra_ec=dict(gop=1),
)


def get_continuous_color(colorscale, intermed):
    """
    Copied from https://stackoverflow.com/a/64655638/1549476
    """
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    [res] = plotly.colors.sample_colorscale(colorscale, intermed)
    return res


def get_color(scale, point):
    import ast

    result = get_continuous_color(scale, point)
    if result.startswith("rgb"):
        result = ast.literal_eval(result[3:])
        result = np.array(list(result))
    else:
        result = matplotlib.colors.to_rgb(result)
        assert len(result) == 3
        result = list(result)
        result = np.array(result) * 255
    result = result.astype(np.uint8)
    return result
