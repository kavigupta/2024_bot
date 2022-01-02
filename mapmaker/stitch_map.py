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
    get_popular_vote_by_voteshare,
    get_state_results,
    get_state_results_by_voteshare,
    number_votes,
)
from .mapper import USABaseMap, USAPresidencyBaseMap, USASenateBaseMap
from .version import version
from .colors import (
    BACKGROUND_RGB,
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
    pop_vote,
    total_turnout,
    scale=SCALE,
    *,
    profile,
    state,
    insets,
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
            profile.vs,
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
                f"{profile.bot_name} v{version} {profile.credit}",
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

    amount = 30 if len(profile.symbol) == 2 else 15

    y += (amount + 10) // 2 + 20

    draw_text(
        draw,
        amount * scale,
        profile.display_popular_vote(pop_vote),
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

    for inset in insets:
        draw_text(
            draw,
            4 * scale,
            [(inset["text"], profile.text_color)],
            inset["x"] * scale,
            (inset["y"] - 2) * scale,
            align="left",
        )

    populated_map.draw_tipping_point(
        draw=draw, scale=scale, profile=profile, text_center=TEXT_CENTER, y=y
    )

    draw_legend(draw, scale, "county", profile=profile)
    if state:
        draw_legend(draw, scale, "state", profile=profile)

    return im


def draw_legend(draw, scale, mode, *, profile):
    if mode == "state":
        legend_y = LEGEND_STARTY_STATE
        legend_x = LEGEND_STARTX_STATE
    else:
        legend_y = LEGEND_STARTY_COUNTY
        legend_x = LEGEND_STARTX_COUNTY

    def process(color):
        if mode == "state":
            color = color.lstrip("#")
            color = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        else:
            color = color.astype(np.int)
        return color

    def add_square(color, text):
        color = process(color)
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
        for color, text in profile.county_legend:
            add_square(color, text)
    else:
        if len(profile.symbol) == 2:
            for color, text in profile.state_legend:
                add_square(color, text)
        else:
            legend_x -= LEGEND_SIZE
            for column, party in enumerate(sorted(profile.symbol)):
                for row, (color, short) in enumerate(
                    zip(profile.state_colors(party), profile.state_symbols_short(party))
                ):
                    color = process(color)
                    if row == 0:
                        draw_text(
                            draw,
                            int(LEGEND_SIZE * 0.8) * scale,
                            [(profile.symbol[party], "rgb" + str(tuple(color)))],
                            (legend_x + column * LEGEND_SIZE + LEGEND_SIZE // 2)
                            * scale,
                            int((legend_y - 0.2 * LEGEND_SIZE) * scale),
                            align="center",
                        )
                    if column == len(profile.symbol) - 1:
                        draw_text(
                            draw,
                            int(LEGEND_SIZE * 0.8) * scale,
                            [(short, profile.text_color)],
                            (legend_x + (column + 1.25) * LEGEND_SIZE) * scale,
                            int((legend_y + (row + 0.9) * LEGEND_SIZE) * scale),
                            align="left",
                        )
                    draw.rectangle(
                        (
                            (legend_x + column * LEGEND_SIZE) * scale,
                            (legend_y + row * LEGEND_SIZE) * scale,
                            (legend_x + (column + 1) * LEGEND_SIZE) * scale,
                            (legend_y + (row + 1) * LEGEND_SIZE) * scale,
                        ),
                        (*color, 255),
                    )


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
    voteshare_by_party = dict(dem=dem_margin / 2 + 0.5, gop=-dem_margin / 2 + 0.5)
    return produce_entire_map_generic(
        data,
        title,
        out_path,
        voteshare_by_party=voteshare_by_party,
        turnout=turnout,
        basemap=basemap,
        year=year,
        profile=profile,
        use_png=use_png,
    )


def produce_entire_map_generic(
    data,
    title,
    out_path,
    *,
    voteshare_by_party,
    turnout,
    basemap,
    year,
    profile=STANDARD_PROFILE,
    use_png=True,
):
    voteshare_by_party_to_map = {
        k: voteshare_by_party[k] * basemap.county_mask(year) for k in voteshare_by_party
    }

    populated_map = basemap.populate(data, voteshare_by_party_to_map, turnout)

    cm, cm_insets = basemap.map_county_margins(
        data["FIPS"], voteshare_by_party=voteshare_by_party_to_map, profile=profile
    )
    sm = basemap.state_map(
        data,
        voteshare_by_party=voteshare_by_party_to_map,
        turnout=turnout,
        profile=profile,
    )
    pop_vote = get_popular_vote_by_voteshare(
        data, voteshare_by_party=voteshare_by_party, turnout=turnout
    )

    fig = sg.SVGFigure("160cm", "65cm")

    counties_svg, states_svg = [tempfile.mktemp(suffix=".svg") for _ in range(2)]
    county_insets_svg = {
        inset: tempfile.mktemp(suffix=".svg") for inset in basemap.insets
    }
    text_mask = tempfile.mktemp(suffix=".png")

    cm.write_image(counties_svg)
    for inset in basemap.insets:
        cm_insets[inset].write_image(county_insets_svg[inset])
    if sm is not None:
        sm.write_image(states_svg)

    # load matpotlib-generated figures
    remove_backgrounds(counties_svg)
    for inset in basemap.insets:
        remove_backgrounds(county_insets_svg[inset])
    if sm is not None:
        remove_backgrounds(states_svg)

    maps = [sg.fromfile(counties_svg).getroot(), *basemap.extra_county_maps]
    for map in maps:
        s = basemap.map_scale
        map.moveto(0, basemap.map_dy, scale_x=s, scale_y=s)

    fig.append(maps)

    inset_text = []

    for inset in basemap.insets:
        s = basemap.map_scale * basemap.insets[inset].scale
        map = sg.fromfile(county_insets_svg[inset]).getroot()
        x = basemap.insets[inset].x_out
        y = basemap.map_dy + basemap.insets[inset].y_out
        map.moveto(x, y, scale_x=s, scale_y=s)
        fig.append([map])
        inset_text.append(dict(x=x+basemap.insets[inset].text_dx, y=y, text=basemap.insets[inset].name))

    if sm is not None:
        states = sg.fromfile(states_svg).getroot()
        states.moveto(
            575,
            200 + 0.5 * basemap.map_dy,
            scale_x=0.5 * basemap.map_scale,
            scale_y=0.5 * basemap.map_scale,
        )
        fig.append([states])

    im = produce_text(
        title,
        populated_map,
        pop_vote,
        total_turnout=number_votes(data, turnout=turnout)
        / number_votes(data, turnout=1),
        profile=profile,
        state=sm is not None,
        insets=inset_text,
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
    _, values = get_state_results_by_voteshare(
        data, voteshare_by_party=voteshare_by_party, turnout=turnout
    )
    result = serialize_output(profile, values)
    return result


def serialize_output(profile, values, always_whole=False):
    result = []
    for value in values:
        parties = sorted(value)
        if len(parties) == 2 and not always_whole:
            result.append(value[parties[0]] - value[parties[1]])
        else:
            whole = " ".join(
                f"{profile.symbol[party]}={value[party]:.2%}" for party in parties
            )
            result.append(whole)
    return result


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
