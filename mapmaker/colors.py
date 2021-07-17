import numpy as np
import matplotlib.colors

import plotly.colors

BACKGROUND = "#222"
BACKGROUND_RGB = "rgb(34, 34, 34)"

COUNTY_SCALE_MARGIN_MAX = 0.8
COUNTY_SCALE_MARGIN_MIN = -0.8

MARGINAL = 0.001

COUNTY_MAX_GOP = "#ff0000"
COUNTY_MIN_GOP = "#ffaaaa"
EVEN = "#ffffff"
COUNTY_MAX_DEM = "#0000ff"
COUNTY_MIN_DEM = "#aaaaff"

COUNTY_COLORSCALE = [
    [0, COUNTY_MAX_GOP],
    [0.5 - MARGINAL, COUNTY_MIN_GOP],
    [0.5, EVEN],
    [0.5 + MARGINAL, COUNTY_MIN_DEM],
    [1.0, COUNTY_MAX_DEM],
]


STATE_GOP_TILT = "#ffcfcf"
STATE_DEM_TILT = "#cfcfff"

STATE_GOP_LEAN = "#ff9d9c"
STATE_DEM_LEAN = "#9d9dff"

STATE_GOP_LIKELY = "#ff6a69"
STATE_DEM_LIKELY = "#6a69ff"

STATE_GOP = "#ff3836"
STATE_DEM = "#3836ff"

TEXT_COLOR = "white"


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
