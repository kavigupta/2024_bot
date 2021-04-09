
import tqdm

from mapmaker.model import Model
from mapmaker.data import all_data

model = Model(all_data(), alpha=0.25)
model.sample("2020 Actual", seed=None, path="images/2020_actual.svg")
model.sample("2020 Pred", seed=None, path="images/2020_pred.svg", correct=False)
model.unbias_predictor()
for i in tqdm.trange(1, 11):
    model.sample(
        f"2024 speculative {i}",
        seed=i,
        path=f"images/2024_speculative_{i}.svg",
    )
