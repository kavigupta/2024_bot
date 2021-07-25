import os
import functools
import pickle

from .data import data_by_year
from .constants import PCA_DIMENSIONS
from .torch_model import DemographicCategoryModel
from .version import version
from .calibrator import calibrate
from .stitch_demographic_map import generate_demographic_map

IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images", version
)


@functools.lru_cache(None)
def get_model(calibrated=False, *, dimensions=PCA_DIMENSIONS):
    model = DemographicCategoryModel(
        data_by_year(), feature_kwargs=dict(dimensions=dimensions)
    )
    if calibrated:
        model = calibrate(model, for_year=2024)
    return model


def get_image(seed, name, *, map_type):
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
        return True, png_path, pkl_path
    stateres = get_model(calibrated=True).sample_map(
        f"2024 scenario {name}",
        year=2024,
        seed=seed,
        path=svg_path,
        map_type=map_type,
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(stateres, f)
    return False, png_path, pkl_path


def get_demographics_image(year, filepath):
    county_demographic_data = get_model(calibrated=False).get_demographics_by_county(
        year=year
    )
    generate_demographic_map(
        data_by_year()[year], county_demographic_data, "sateohusaoteh", filepath
    )
