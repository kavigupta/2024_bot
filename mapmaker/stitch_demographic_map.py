import os
import re
import tempfile

from PIL import Image, ImageDraw
import numpy as np

import svgutils.transform as sg
from cairosvg import svg2png

from .aggregation import (
    get_electoral_vote,
    get_popular_vote,
    get_state_results,
    calculate_tipping_point,
    number_votes,
)
from .mapper import map_county_demographics
from .version import version
from .colors import (
    BACKGROUND_RGB,
    TEXT_COLOR,
    STATE_GOP,
    STATE_DEM,
    STATE_GOP_BATTLEGROUND,
    STATE_DEM_BATTLEGROUND,
    STATE_GOP_CLOSE,
    STATE_DEM_CLOSE,
    COUNTY_MAX_VAL,
    COUNTY_MIN_VAL,
)
from .text import draw_text

LEFT_MARGIN = 50
RIGHT_MARGIN = 25
TOP_MARGIN = 55
BOTTOM_MARGIN = 15
TEXT_CENTER = 760

FIRST_LINE = 110

LEGEND_STARTY_COUNTY = 170
LEGEND_STARTX_COUNTY = 40
LEGEND_STARTY_STATE = 350
LEGEND_STARTX_STATE = 870
LEGEND_SIZE = 10

SCALE = 4

def generate_demographic_map(data, demographic_values, title, out_path):
    cm = map_county_demographics(data, demographic_values=demographic_values)

    fig = sg.SVGFigure("160cm", "65cm")

    counties_svg, states_svg = [tempfile.mktemp(suffix=".svg") for _ in range(2)]
    text_mask = tempfile.mktemp(suffix=".png")

    cm.write_image(counties_svg)

    # load matpotlib-generated figures
    remove_backgrounds(counties_svg)

    fig.append([sg.fromfile(counties_svg).getroot()])

    # im = produce_text(
    #     title,
    #     dem_ec,
    #     dem_ec_close,
    #     gop_ec,
    #     gop_ec_close,
    #     pop_vote_margin,
    #     tipping_point_state,
    #     tipping_point_margin,
    #     total_turnout=number_votes(data, turnout=turnout)
    #     / number_votes(data, turnout=1),
    # )
    # im.save(text_mask)
    # with open(text_mask, "rb") as f:
    #     fig.append(sg.ImageElement(f, 950, 450))
    fig.save(out_path)
    add_background_back(out_path)
    with open(out_path) as f:
        svg2png(
            bytestring=f.read(), write_to=out_path.replace(".svg", ".png"), scale=SCALE
        )
    os.remove(out_path)
