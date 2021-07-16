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
from .mapper import map_county_margins, state_map
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
    COUNTY_COLORSCALE,
    COUNTY_SCALE_MARGIN_MAX,
    COUNTY_SCALE_MARGIN_MIN,
    get_color,
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


def produce_text(
    title,
    dem_ec,
    dem_ec_close,
    gop_ec,
    gop_ec_close,
    pop_vote_margin,
    tipping_point_state,
    tipping_point_margin,
    total_turnout,
    scale=SCALE,
):
    im = Image.new(mode="RGBA", size=(950 * scale, 450 * scale))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, *im.size), (0, 0, 0, 0))

    draw_text(
        draw, 45 * scale, [(title, TEXT_COLOR)], LEFT_MARGIN * scale, TOP_MARGIN * scale
    )
    draw_text(
        draw,
        15 * scale,
        [("@bot_2024", TEXT_COLOR)],
        (950 - RIGHT_MARGIN) * scale,
        TOP_MARGIN * scale,
        align="right",
    )
    draw_text(
        draw,
        10 * scale,
        [
            (
                f"bot_2024 v{version} by @notkavi and @lxeagle17 with data from @mill226",
                TEXT_COLOR,
            )
        ],
        (950 - RIGHT_MARGIN) * scale,
        (450 - BOTTOM_MARGIN) * scale,
        align="right",
    )

    y = FIRST_LINE

    draw_text(
        draw,
        40 * scale,
        [(str(dem_ec), STATE_DEM), (" - ", TEXT_COLOR), (str(gop_ec), STATE_GOP)],
        TEXT_CENTER * scale,
        y * scale,
        align=("center", 1),
    )

    y += 15 // 2 + 20

    draw_text(
        draw,
        15 * scale,
        [
            ("Close: ", TEXT_COLOR),
            (str(dem_ec_close), STATE_DEM_CLOSE),
            (" - ", TEXT_COLOR),
            (str(gop_ec_close), STATE_GOP_CLOSE),
        ],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    y += 40 // 2 + 20

    draw_text(
        draw,
        30 * scale,
        [
            (f"D+{pop_vote_margin:.2%}", STATE_DEM)
            if pop_vote_margin > 0
            else (f"R+{-pop_vote_margin:.2%}", STATE_GOP)
        ],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    y += 10 // 2 + 20

    draw_text(
        draw,
        10 * scale,
        [(f"Total Turnout: {total_turnout:.0%}", TEXT_COLOR)],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    y += 20

    tipping_point_str = None
    tipping_point_color = None

    if tipping_point_margin > 0:
        tipping_point_str = f"{tipping_point_state} D+{tipping_point_margin:.2%}"
        tipping_point_color = STATE_DEM_CLOSE
    else:
        tipping_point_str = f"{tipping_point_state} R+{-tipping_point_margin:.2%}"
        tipping_point_color = STATE_GOP_CLOSE

    draw_text(
        draw,
        10 * scale,
        [("Tipping Point: ", TEXT_COLOR), (tipping_point_str, tipping_point_color)],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    draw_legend(draw, scale, "county")
    draw_legend(draw, scale, "state")

    return im


def draw_legend(draw, scale, mode):
    if mode == "state":
        legend_y = LEGEND_STARTY_STATE
        legend_x = LEGEND_STARTX_STATE
    else:
        legend_y = LEGEND_STARTY_COUNTY
        legend_x = LEGEND_STARTX_COUNTY

    def add_square(color, text):
        if mode == "state":
            color = color.lstrip("#")
            color = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        else:
            color = color.astype(np.int)
        nonlocal legend_y
        draw.rectangle(
            (
                legend_x * scale,
                legend_y * scale,
                (legend_x + LEGEND_SIZE) * scale,
                (legend_y + LEGEND_SIZE) * scale,
            ),
            (*color, 255),
        )
        legend_y += LEGEND_SIZE
        if mode == "state":
            draw_text(
                draw,
                int(LEGEND_SIZE * 0.8) * scale,
                [(text, "rgb" + str(tuple(color)))],
                (legend_x + LEGEND_SIZE * 1.25) * scale,
                int((legend_y - LEGEND_SIZE * 0.3) * scale),
                align="left",
            )
        else:
            draw_text(
                draw,
                int(LEGEND_SIZE * 0.8) * scale,
                [(text, "rgb" + str(tuple(color)))],
                (legend_x - LEGEND_SIZE // 3) * scale,
                int((legend_y - LEGEND_SIZE * 0.3) * scale),
                align="right",
            )

    if mode == "county":
        for margin in np.arange(-0.8, 0.8 + 1e-1, 0.2):
            add_square(
                get_color(
                    COUNTY_COLORSCALE,
                    (margin - COUNTY_SCALE_MARGIN_MIN)
                    / (COUNTY_SCALE_MARGIN_MAX - COUNTY_SCALE_MARGIN_MIN),
                ),
                f"R+{-margin * 100:.0f}"
                if margin < -0.001
                else f"D+{margin * 100:.0f}"
                if margin > 0.001
                else "Even",
            )
    else:
        state_buckets = ["> R+5", "R+1 - R+5", "< R+1", "< D+1", "D+1 - D+5", "> D+5"]
        state_colors = [
            STATE_GOP,
            STATE_GOP_BATTLEGROUND,
            STATE_GOP_CLOSE,
            STATE_DEM_CLOSE,
            STATE_DEM_BATTLEGROUND,
            STATE_DEM,
        ]
        for margin_text, color in zip(state_buckets, state_colors):
            add_square(color, margin_text)


def generate_map(data, title, out_path, *, dem_margin, turnout):
    dem_ec, gop_ec = get_electoral_vote(data, dem_margin=dem_margin, turnout=turnout)
    dem_ec_safe, gop_ec_safe = get_electoral_vote(
        data, dem_margin=dem_margin, turnout=turnout, only_nonclose=True
    )

    tipping_point_state, tipping_point_margin = calculate_tipping_point(
        data, dem_margin=dem_margin, turnout=turnout
    )

    dem_ec_close, gop_ec_close = dem_ec - dem_ec_safe, gop_ec - gop_ec_safe
    assert dem_ec_close >= 0 and gop_ec_close >= 0
    cm = map_county_margins(data, dem_margin=dem_margin)
    sm = state_map(data, dem_margin=dem_margin, turnout=turnout)
    pop_vote_margin = get_popular_vote(data, dem_margin=dem_margin, turnout=turnout)

    fig = sg.SVGFigure("160cm", "65cm")

    counties_svg, states_svg = [tempfile.mktemp(suffix=".svg") for _ in range(2)]
    text_mask = tempfile.mktemp(suffix=".png")

    cm.write_image(counties_svg)
    sm.write_image(states_svg)

    # load matpotlib-generated figures
    remove_backgrounds(counties_svg)
    remove_backgrounds(states_svg)

    fig.append([sg.fromfile(counties_svg).getroot()])
    states = sg.fromfile(states_svg).getroot()
    states.moveto(575, 200, scale_x=0.5, scale_y=0.5)
    fig.append([states])

    im = produce_text(
        title,
        dem_ec,
        dem_ec_close,
        gop_ec,
        gop_ec_close,
        pop_vote_margin,
        tipping_point_state,
        tipping_point_margin,
        total_turnout=number_votes(data, turnout=turnout)
        / number_votes(data, turnout=1),
    )
    im.save(text_mask)
    with open(text_mask, "rb") as f:
        fig.append(sg.ImageElement(f, 950, 450))
    fig.save(out_path)
    add_background_back(out_path)
    with open(out_path) as f:
        svg2png(
            bytestring=f.read(), write_to=out_path.replace(".svg", ".png"), scale=SCALE
        )
    os.remove(out_path)
    return get_state_results(data, dem_margin=dem_margin, turnout=turnout)


def remove_backgrounds(path):
    with open(path) as f:
        contents = f.read()
    contents = re.sub(r'<rect x="0" y="0" [^/]*"/>', "", contents)
    contents = re.sub(
        r"<rect[^/]*" + re.escape(re.escape(BACKGROUND_RGB)) + "[^/]*/>", "", contents
    )
    contents = re.sub('<g class="layer land">.*?</g>', "", contents)
    contents = re.sub('<g class="layer subunits">.*?</g>', "", contents)
    with open(path, "w") as f:
        f.write(contents)


def add_background_back(path):
    with open(path) as f:
        contents = f.read()
    start_content = contents.index("<g>")
    insert_rect = f'<rect x="0" y="0" width="950" height="450" style="fill: {BACKGROUND_RGB}; fill-opacity: 1"/>'
    contents = contents[:start_content] + insert_rect + contents[start_content:]
    contents = contents.replace(
        'version="1.1"', 'version="1.1" width="950" height="450"'
    )
    with open(path, "w") as f:
        f.write(contents)
