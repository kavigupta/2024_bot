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
    model = Model(all_data(), alpha=0.1)
    model.unbias_predictor()
    return model


def get_image(seed, name):
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
        f"2024 scenario: {name}",
        seed=seed,
        path=svg_path,
    )
    return False, png_path


def sample_word():
    number = random.randint(0, 2 ** 16 - 1)
    return number, " ".join(id_to_words(number))


def sample_image():
    number, word = sample_word()
    exists, path = get_image(number, word)
    if exists:
        return sample_image()
    return word, path
