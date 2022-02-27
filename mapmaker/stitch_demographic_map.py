import os
import tempfile

import tqdm

import svgutils.transform as sg
from cairosvg import svg2png

from mapmaker.colors import STANDARD_PROFILE

from .mapper import USAPresidencyBaseMap
from .stitch_map import remove_backgrounds, add_background_back

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
    print("generating", title)
    num_demos = demographic_values.shape[1]
    # import IPython; IPython.embed()
    cms = [
        USAPresidencyBaseMap().map_county_demographics(
            data["FIPS"],
            demographic_values=demographic_values[:, i],
            profile=STANDARD_PROFILE,
        )
        for i in tqdm.trange(num_demos)
    ]

    fig = sg.SVGFigure("160cm", "65cm")

    counties_svgs = [tempfile.mktemp(suffix=".svg") for _ in range(num_demos)]
    # text_mask = tempfile.mktemp(suffix=".png")

    for cm, counties_svg in zip(cms, counties_svgs):
        cm.write_image(counties_svg)
        remove_backgrounds(counties_svg, STANDARD_PROFILE)

    cms = [
        sg.fromfile(counties_svg).getroot() for counties_svg in tqdm.tqdm(counties_svgs)
    ]

    for i, cm in list(enumerate(cms)):
        row, column = i // 4, i % 4
        cm.moveto(column * 200, row * 100, scale_x=0.2, scale_y=0.2)

    fig.append(cms)

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
    add_background_back(out_path, STANDARD_PROFILE)
    with open(out_path) as f:
        svg2png(
            bytestring=f.read(), write_to=out_path.replace(".svg", ".png"), scale=SCALE
        )
    os.remove(out_path)
