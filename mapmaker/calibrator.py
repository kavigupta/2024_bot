import numpy as np
import tqdm

from .data import ec


def bias(preds):
    dems = preds > 0
    gop = preds < 0
    ecs = np.array(ec())[:, 0]
    return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()


def calibrate(model, *, for_year, target_pv_spread_90=17.5e-2):
    low_alpha, high_alpha = 0, 1
    while True:
        mid_alpha = (low_alpha + high_alpha) / 2
        model = model.with_alpha(mid_alpha)
        state_preds, pv = model.family_of_predictions(year=for_year)
        pv_spread_90 = np.percentile(pv, 95) - np.percentile(pv, 5)
        print(f"Alpha: {mid_alpha:.4f}")
        print(f"Spread: {pv_spread_90:.2%}")
        if abs(pv_spread_90 - target_pv_spread_90) < 0.5e-2:
            break
        if pv_spread_90 > target_pv_spread_90:
            high_alpha = mid_alpha
        else:
            low_alpha = mid_alpha

    print(
        f"Without correction, democrats win the EC {bias(state_preds):.2%} of the time"
    )

    bias_values = np.arange(-0.1, 0.1, 0.01e-2)
    biases = np.array([bias(state_preds + x) for x in tqdm.tqdm(bias_values)])

    idx = np.argmin(np.abs(biases - 0.5))
    best_bias = bias_values[idx]
    print(
        f"Best bias would be: {best_bias:.2%}, which gives democrats an EC win {biases[idx]:.0%} of the time"
    )
    # model = model.with_predictor(model.predictor.with_bias(best_bias))
    return model
