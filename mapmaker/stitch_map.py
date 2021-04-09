import os
import re
import tempfile

import svgutils.transform as sg
from cairosvg import svg2png

from .processing import get_electoral_vote, get_popular_vote
from .mapper import county_map, state_map
from .version import version


def generate_map(data, dem_margin, title, out_path):
    dem_ec, gop_ec = get_electoral_vote(data, dem_margin)
    cm = county_map(data, dem_margin)
    sm = state_map(data, dem_margin)
    pop_vote_margin = get_popular_vote(data, dem_margin)

    fig = sg.SVGFigure("160cm", "65cm")

    counties_svg, states_svg = [tempfile.mktemp(suffix=".svg") for _ in range(2)]

    cm.write_image(counties_svg)
    sm.write_image(states_svg)

    # load matpotlib-generated figures
    remove_backgrounds(counties_svg)
    remove_backgrounds(states_svg)

    fig1 = sg.fromfile(counties_svg)
    fig2 = sg.fromfile(states_svg)

    # get the plot objects
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    plot2.moveto(575, 200, scale_x=0.5, scale_y=0.5)
    fig.append([plot1, plot2])
    fig.append(
        [sg.TextElement(125, 55, title, size=45, color="#fff", font="Cantarell")]
    )
    fig.append(
        [sg.TextElement(800, 55, "@bot_2024", size=15, color="#fff", font="Cantarell")]
    )

    ecx, ecy = 675, 150
    fig.append(
        [sg.TextElement(ecx, ecy, str(dem_ec), size=40, color="#88f", font="Cantarell")]
    )
    fig.append(
        [sg.TextElement(ecx + 80, ecy, "-", size=40, color="#fff", font="Cantarell")]
    )
    fig.append(
        [
            sg.TextElement(
                ecx + 105, ecy, str(gop_ec), size=40, color="#f88", font="Cantarell"
            )
        ]
    )
    pvx, pvy = ecx + 20, ecy + 60
    if pop_vote_margin < 0:
        fig.append(
            [
                sg.TextElement(
                    pvx,
                    pvy,
                    f"R+{-pop_vote_margin:.2%}",
                    size=30,
                    color="#f88",
                    font="Cantarell",
                )
            ]
        )
    else:
        fig.append(
            [
                sg.TextElement(
                    pvx,
                    pvy,
                    f"D+{pop_vote_margin:.2%}",
                    size=30,
                    color="#88f",
                    font="Cantarell",
                )
            ]
        )
    fig.append(
        [
            sg.TextElement(
                820,
                435,
                f"2024bot v{version}",
                size=10,
                color="#fff",
                font="Cantarell",
            )
        ]
    )

    fig.save(out_path)
    add_background_back(out_path)
    with open(out_path) as f:
        svg2png(bytestring=f.read(), write_to=out_path.replace(".svg", ".png"))


def remove_backgrounds(path):
    with open(path) as f:
        contents = f.read()
    contents = re.sub(r'<rect x="0" y="0" [^/]*"/>', "", contents)
    contents = re.sub(r"<rect[^/]*rgb\(34, 34, 34\)[^/]*/>", "", contents)
    with open(path, "w") as f:
        f.write(contents)


def add_background_back(path):
    with open(path) as f:
        contents = f.read()
    start_content = contents.index("<g>")
    insert_rect = '<rect x="0" y="0" width="900" height="450" style="fill: rgb(34, 34, 34); fill-opacity: 1"/>'
    contents = contents[:start_content] + insert_rect + contents[start_content:]
    contents = contents.replace(
        'version="1.1"', 'version="1.1" width="900" height="450"'
    )
    with open(path, "w") as f:
        f.write(contents)
