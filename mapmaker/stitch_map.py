import os
import re
import tempfile

from PIL import Image, ImageDraw

import svgutils.transform as sg
from cairosvg import svg2png

from .aggregation import get_electoral_vote, get_popular_vote, get_state_results
from .mapper import county_map, state_map
from .version import version
from .colors import (
    BACKGROUND_RGB,
    TEXT_COLOR,
    STATE_DEM,
    STATE_GOP,
    STATE_DEM_CLOSE,
    STATE_GOP_CLOSE,
)
from .text import draw_text

LEFT_MARGIN = 50
RIGHT_MARGIN = 25
TOP_MARGIN = 55
BOTTOM_MARGIN = 15
TEXT_CENTER = 760

FIRST_LINE = 150


def produce_text(
    title, dem_ec, dem_ec_close, gop_ec, gop_ec_close, pop_vote_margin, scale=5
):
    im = Image.new(mode="RGBA", size=(900 * scale, 450 * scale))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, *im.size), (0, 0, 0, 0))

    draw_text(
        draw, 45 * scale, [(title, TEXT_COLOR)], LEFT_MARGIN * scale, TOP_MARGIN * scale
    )
    draw_text(
        draw,
        15 * scale,
        [("@bot_2024", TEXT_COLOR)],
        (900 - RIGHT_MARGIN) * scale,
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
        (900 - RIGHT_MARGIN) * scale,
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
    return im


def generate_map(data, dem_margin, title, out_path):
    dem_ec, gop_ec = get_electoral_vote(data, dem_margin)
    dem_ec_safe, gop_ec_safe = get_electoral_vote(data, dem_margin, only_nonclose=True)
    dem_ec_close, gop_ec_close = dem_ec - dem_ec_safe, gop_ec - gop_ec_safe
    assert dem_ec_close >= 0 and gop_ec_close >= 0
    cm = county_map(data, dem_margin)
    sm = state_map(data, dem_margin)
    pop_vote_margin = get_popular_vote(data, dem_margin)

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
        title, dem_ec, dem_ec_close, gop_ec, gop_ec_close, pop_vote_margin
    )
    im.save(text_mask)
    with open(text_mask, "rb") as f:
        fig.append(sg.ImageElement(f, 900, 450))
    fig.save(out_path)
    add_background_back(out_path)
    with open(out_path) as f:
        svg2png(bytestring=f.read(), write_to=out_path.replace(".svg", ".png"), scale=5)
    return get_state_results(data, dem_margin)


def remove_backgrounds(path):
    with open(path) as f:
        contents = f.read()
    contents = re.sub(r'<rect x="0" y="0" [^/]*"/>', "", contents)
    contents = re.sub(
        r"<rect[^/]*" + re.escape(re.escape(BACKGROUND_RGB)) + "[^/]*/>", "", contents
    )
    with open(path, "w") as f:
        f.write(contents)


def add_background_back(path):
    with open(path) as f:
        contents = f.read()
    start_content = contents.index("<g>")
    insert_rect = f'<rect x="0" y="0" width="900" height="450" style="fill: {BACKGROUND_RGB}; fill-opacity: 1"/>'
    contents = contents[:start_content] + insert_rect + contents[start_content:]
    contents = contents.replace(
        'version="1.1"', 'version="1.1" width="900" height="450"'
    )
    with open(path, "w") as f:
        f.write(contents)
