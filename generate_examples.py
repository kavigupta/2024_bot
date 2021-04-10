
import tqdm

from mapmaker.model import Model
from mapmaker.data import all_data
from mapmaker.generate_image import get_model, get_image

model = get_model()
model.sample("2020 Actual", seed=None, path="images/2020_actual.svg")
model.sample("2020 Pred", seed=None, path="images/2020_pred.svg", correct=False)
model.unbias_predictor()
for i in tqdm.trange(1, 11):
    get_image(i, i)
