import tqdm

from mapmaker.generate_image import get_model, get_image

model = get_model(calibrated=False)
model.sample_map(
    "2020 Actual",
    seed=None,
    path="images/2020_actual.svg",
    year=2020,
)
model.sample_map(
    "2020 Pred",
    seed=None,
    path="images/2020_pred.svg",
    correct=False,
    year=2020,
)
model.sample_map(
    "2024 Pred",
    seed=None,
    path="images/2024_pred.svg",
    correct=False,
    year=2024,
)
model.sample_map(
    "2024 Pred Corrected",
    seed=None,
    path="images/2024_pred_corrected.svg",
    year=2024,
)
for i in tqdm.trange(10, 20):
    get_image(i, i)
