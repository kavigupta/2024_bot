import tqdm

from mapmaker.model import Model
from mapmaker.generate_image import get_model, get_image

model = get_model(unbias=False)
model.sample_map(
    "2020 Actual",
    seed=None,
    path="images/2020_actual.svg",
    adjust=False,
    year=2020,
)
model.sample_map(
    "2020 Pred",
    seed=None,
    path="images/2020_pred.svg",
    adjust=False,
    correct=False,
    year=2020,
)
model.sample_map(
    "2024 Pred",
    seed=None,
    path="images/2024_pred.svg",
    correct=False,
    adjust=False,
    year=2024,
)
model.sample_map(
    "2024 Pred Corrected",
    seed=None,
    path="images/2024_pred_corrected.svg",
    year=2024,
)
for i in tqdm.trange(1, 11):
    get_image(i, i)
