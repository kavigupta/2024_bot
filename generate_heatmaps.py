import tqdm

from mapmaker.generate_image import get_model, get_demographics_image

model = get_model(calibrated=False)
for y in (
    2024,
    2012,
    2016,
    2020,
):
    get_demographics_image(y, "images/demographic_" + str(y) + ".svg")
