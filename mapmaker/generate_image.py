import os
import re
import functools
import random

import requests

from permacache import permacache
from word_identifiers import id_to_words

from .data import all_data
from .model import Model
from .version import version

IMAGE_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "images", version
)


@functools.lru_cache(None)
def get_model():
    model = Model(all_data(), alpha=0.1, feature_kwargs=dict(pca=30))
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
    if os.path.exists(png_path):
        return True, png_path
    get_model().sample(

        f"2024 scenario {name}",
        data=all_data(demographic_projection=True)
        seed=seed,
        path=svg_path,
    )
    return False, png_path
