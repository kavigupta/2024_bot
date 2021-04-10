
import tqdm

from mapmaker.model import Model
from mapmaker.data import all_data
from mapmaker.generate_image import get_model, get_image

model = get_model(unbias=False)
model.sample("2020 Actual", seed=None, path="images/2020_actual.svg", data=all_data())
model.sample("2020 Pred", seed=None, path="images/2020_pred.svg", correct=False, data=all_data())
model.sample("2024 Pred", seed=None, path="images/2024_pred.svg", correct=False, data=all_data(demographic_projection=True))
model.sample("2024 Pred Corrected", seed=None, path="images/2024_pred_corrected.svg", data=all_data(demographic_projection=True))
model.unbias_predictor()
for i in tqdm.trange(1, 11):
    get_image(i, i)
