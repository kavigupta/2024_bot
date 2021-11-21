import os
import functools
import pickle

from mapmaker.mapper import USAPresidencyBaseMap
from .specials.gondor import generate_gondor_map

from .data import data_by_year
from .constants import NUM_DEMOGRAPHICS, PCA_DIMENSIONS
from .torch_model import DemographicCategoryModel
from .version import version
from .calibrator import calibrate
from .stitch_demographic_map import generate_demographic_map
from .alternate_universe import generate_alternate_universe_map

IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images", version
)


@functools.lru_cache(None)
def get_model(
    calibrated=False, *, dimensions=PCA_DIMENSIONS, num_demographics=NUM_DEMOGRAPHICS
):
    model = DemographicCategoryModel(
        data_by_year(),
        feature_kwargs=dict(dimensions=dimensions),
        num_demographics=num_demographics,
    )
    if calibrated:
        model = calibrate(USAPresidencyBaseMap(), model, for_year=2024)
    return model


def get_image(seed, name, *, basemap):
    name = str(name)
    try:
        os.makedirs(IMAGE_FOLDER)
    except FileExistsError:
        pass

    filename = name.replace(" ", "-")

    svg_path = f"{IMAGE_FOLDER}/{filename}.svg"
    png_path = f"{IMAGE_FOLDER}/{filename}.png"
    pkl_path = f"{IMAGE_FOLDER}/{filename}.pkl"
    if os.path.exists(png_path):
        return png_path, pkl_path
    stateres = get_model(calibrated=True).sample_map(
        f"2024 scenario {name}",
        year=2024,
        seed=seed,
        path=svg_path,
        basemap=basemap,
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(stateres, f)
    return png_path, pkl_path


def get_althistory_image(seed, prefix="images/alternate-universes"):
    path = f"{prefix}/{seed}.svg"
    generate_alternate_universe_map(seed, f"Alternate Universe {seed}", path)
    return path.replace(".svg", ".png"), path.replace(".svg", ".pkl")


def get_gondor_image(seed, prefix="images/gondor"):
    path = f"{prefix}/{seed}.svg"
    generate_gondor_map(
        seed,
        f"Gondor Scenario {seed}" if seed is not None else "Gondor 2020 Actual",
        path,
    )
    return path.replace(".svg", ".png"), path.replace(".svg", ".pkl")


def get_demographics_image(year, filepath):
    county_demographic_data = get_model(calibrated=False).get_demographics_by_county(
        year=year
    )
    generate_demographic_map(
        data_by_year()[year], county_demographic_data, "sateohusaoteh", filepath
    )
