import os
import functools
import pickle

from .data import data_by_year
from .model import Model
from .version import version
from .calibrator import calibrate

IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images", version
)


@functools.lru_cache(None)
def get_model(calibrated=False):
    model = Model(data_by_year(), feature_kwargs=dict(dimensions=19))
    if calibrated:
        model = calibrate(model, for_year=2024)
    return model


def get_image(seed, name):
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
    stateres = get_model(unbias=True).sample_map(
        f"2024 scenario {name}",
        year=2024,
        seed=seed,
        path=svg_path,
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(stateres, f)
    return False, png_path, pkl_path
