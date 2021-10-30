import numpy as np
import tqdm

from .constants import TARGET_PV_SPREAD_50


def bias(basemap, preds):
    dems = preds > 0
    gop = preds < 0
    ecs = np.array(basemap.electoral_votes)[:, 0]
    return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()


def calibrate(basemap, model, *, for_year, pv_spread=TARGET_PV_SPREAD_50):
    low_alpha, high_alpha = 0, 2
    while True:
        mid_alpha = (low_alpha + high_alpha) / 2
        model = model.with_alpha(mid_alpha)
        _, state_preds, pv = model.family_of_predictions(year=for_year, basemap=basemap)
        pv_spread_50 = np.percentile(pv, 75) - np.percentile(pv, 25)
        print(f"Alpha: {mid_alpha:.4f}")
        print(f"Spread: {pv_spread_50:.2%}")
        if abs(pv_spread_50 - pv_spread) < 0.5e-2:
            break
        if pv_spread_50 > pv_spread:
            high_alpha = mid_alpha
        else:
            low_alpha = mid_alpha

    print(
        f"Without correction, democrats win the EC {bias(basemap, state_preds):.2%} of the time"
    )

    bias_values = np.arange(-0.1, 0.1, 0.01e-2)
    biases = np.array([bias(basemap, state_preds + x) for x in tqdm.tqdm(bias_values)])

    idx = np.argmin(np.abs(biases - 0.5))
    best_bias = bias_values[idx]
    print(
        f"Best bias would be: {best_bias:.2%}, which gives democrats an EC win {biases[idx]:.0%} of the time"
    )
    # model = model.with_predictor(model.predictor.with_bias(best_bias))
    return model
