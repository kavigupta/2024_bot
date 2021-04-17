import os
import functools
import pickle

from .data import all_data
from .model import Model
from .version import version

IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images", version
)


@functools.lru_cache(None)
def get_model(unbias=False):
    model = Model(
        all_data(),
        all_data(demographic_projection=True),
        alpha=0.05,
        feature_kwargs=dict(pca=24),
    )
    if unbias:
        model.unbias_predictor()
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
        data=all_data(demographic_projection=True),
        seed=seed,
        path=svg_path,
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(stateres, f)
    return False, png_path, pkl_path
