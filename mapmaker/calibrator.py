import numpy as np
import tqdm

from .data import ec


def bias(preds):
    dems = preds > 0
    gop = preds < 0
    ecs = np.array(ec())[:, 0]
    return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()


def unbias_predictor(model, *, for_year):
    state_preds = model.family_of_predictions(year=for_year)

    print(
        f"Without correction, democrats win the EC {bias(state_preds):.2%} of the time"
    )

    bias_values = np.arange(-0.02, -0.005, 0.001)
    biases = np.array([bias(state_preds + x) for x in tqdm.tqdm(bias_values)])

    idx = np.argmin(np.abs(biases - 0.5))
    best_bias = bias_values[idx]
    print(
        f"Computed best bias: {best_bias:.2%}, which gives democrats an EC win {biases[idx]:.0%} of the time"
    )
    model.predictor = model.predictor.with_bias(best_bias)
