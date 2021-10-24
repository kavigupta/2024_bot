import os
import re
import tempfile

from PIL import Image, ImageDraw
import us
import numpy as np

import svgutils.transform as sg
from cairosvg import svg2png

from .aggregation import (
    get_electoral_vote,
    get_senate_vote,
    get_popular_vote,
    get_state_results,
    calculate_tipping_point,
    number_votes,
)
from .mapper import USABaseMap, USAPresidencyBaseMap, USASenateBaseMap
from .version import version
from .colors import (
    BACKGROUND_RGB,
    COUNTY_SCALE_MARGIN_MAX,
    COUNTY_SCALE_MARGIN_MIN,
    STANDARD_PROFILE,
    get_color,
)
from .senate import senate_2022
from .text import draw_text

LEFT_MARGIN = 50
RIGHT_MARGIN = 25
TOP_MARGIN = 55
BOTTOM_MARGIN = 10
TEXT_CENTER = 760

FIRST_LINE = 110

LEGEND_STARTY_COUNTY = 170
LEGEND_STARTX_COUNTY = 40
LEGEND_STARTY_STATE = 330
LEGEND_STARTX_STATE = 885

LEGEND_SIZE = 10

SCALE = 4


def produce_text(
    title,
    populated_map,
    pop_vote_margin,
    total_turnout,
    scale=SCALE,
    *,
    profile,
):
    im = Image.new(mode="RGBA", size=(950 * scale, 450 * scale))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, *im.size), (0, 0, 0, 0))

    if profile.name == None:
        title_start = TOP_MARGIN
        title_scale = 45
    else:
        title_start = TOP_MARGIN - 20
        title_scale = 35

    draw_text(
        draw,
        title_scale * scale,
        [(title, profile.text_color)],
        LEFT_MARGIN * scale,
        title_start * scale,
    )
    if profile.name != None:
        draw_text(
            draw,
            15 * scale,
            [
                (profile.name["dem"] + " Party", profile.state_safe("dem")),
                (" vs. ", profile.text_color),
                (profile.name["gop"] + " Party", profile.state_safe("gop")),
            ],
            (LEFT_MARGIN) * scale,
            (title_start + 25) * scale,
        )
    draw_text(
        draw,
        15 * scale,
        [(f"@{profile.bot_name}", profile.text_color)],
        (950 - RIGHT_MARGIN) * scale,
        TOP_MARGIN * scale,
        align="right",
    )
    draw_text(
        draw,
        10 * scale,
        [
            (
                f"{profile.bot_name} v{version} by @notkavi and @lxeagle17 with data from @mill226",
                profile.text_color,
            )
        ],
        (950 - RIGHT_MARGIN) * scale,
        (450 - BOTTOM_MARGIN) * scale,
        align="right",
    )

    y = FIRST_LINE

    y = populated_map.draw_topline(
        draw=draw, scale=scale, profile=profile, text_center=TEXT_CENTER, y=y
    )

    y += 40 // 2 + 20

    draw_text(
        draw,
        30 * scale,
        [
            (
                f"{profile.symbol['dem']}+{pop_vote_margin:.2%}",
                profile.state_safe("dem"),
            )
            if pop_vote_margin > 0
            else (
                f"{profile.symbol['gop']}+{-pop_vote_margin:.2%}",
                profile.state_safe("gop"),
            )
        ],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    y += 10 // 2 + 20

    draw_text(
        draw,
        10 * scale,
        [(f"Total Turnout: {total_turnout:.0%}", profile.text_color)],
        TEXT_CENTER * scale,
        y * scale,
        align=("center"),
    )

    y += 20

    populated_map.draw_tipping_point(
        draw=draw, scale=scale, profile=profile, text_center=TEXT_CENTER, y=y
    )

    draw_legend(draw, scale, "county", profile=profile)
    draw_legend(draw, scale, "state", profile=profile)

    return im


def draw_legend(draw, scale, mode, *, profile):
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
                    profile.county_colorscale,
                    (margin - COUNTY_SCALE_MARGIN_MIN)
                    / (COUNTY_SCALE_MARGIN_MAX - COUNTY_SCALE_MARGIN_MIN),
                ),
                f"{profile.symbol['gop']}+{-margin * 100:.0f}"
                if margin < -0.001
                else f"{profile.symbol['dem']}+{margin * 100:.0f}"
                if margin > 0.001
                else "Even",
            )
    else:
        state_buckets = [
            f"> {profile.symbol['gop']}+7",
            f"{profile.symbol['gop']}+3 - {profile.symbol['gop']}+7",
            f"{profile.symbol['gop']}+1 - {profile.symbol['gop']}+3",
            f"< {profile.symbol['gop']}+1",
            f"< {profile.symbol['dem']}+1",
            f"{profile.symbol['dem']}+1 - {profile.symbol['dem']}+3",
            f"{profile.symbol['dem']}+3 - {profile.symbol['dem']}+7",
            f"> {profile.symbol['dem']}+7",
        ]
        state_colors = [
            profile.state_safe("gop"),
            profile.state_likely("gop"),
            profile.state_lean("gop"),
            profile.state_tilt("gop"),
            profile.state_tilt("dem"),
            profile.state_lean("dem"),
            profile.state_likely("dem"),
            profile.state_safe("dem"),
        ]
        for margin_text, color in zip(state_buckets, state_colors):
            add_square(color, margin_text)


def produce_entire_map(
    data,
    title,
    out_path,
    *,
    dem_margin,
    turnout,
    basemap,
    year,
    profile=STANDARD_PROFILE,
    use_png=True,
):
    dem_margin_to_map = dem_margin * basemap.county_mask(year)

    populated_map = basemap.populate(data, dem_margin_to_map, turnout)

    cm = basemap.map_county_margins(
        data["FIPS"], dem_margin=dem_margin_to_map, profile=profile
    )
    sm = basemap.state_map(
        data, dem_margin=dem_margin_to_map, turnout=turnout, profile=profile
    )
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
    fig.append(
        [
            sg.fromfile(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../stateboundariesinternal.svg",
                )
            ).getroot()
        ]
    )
    states = sg.fromfile(states_svg).getroot()
    states.moveto(575, 200, scale_x=0.5, scale_y=0.5)
    fig.append([states])

    im = produce_text(
        title,
        populated_map,
        pop_vote_margin,
        total_turnout=number_votes(data, turnout=turnout)
        / number_votes(data, turnout=1),
        profile=profile,
    )
    im.save(text_mask)
    with open(text_mask, "rb") as f:
        fig.append(sg.ImageElement(f, 950, 450))
    fig.save(out_path)
    add_background_back(out_path)
    if use_png:
        with open(out_path) as f:
            svg2png(
                bytestring=f.read(),
                write_to=out_path.replace(".svg", ".png"),
                scale=SCALE,
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
