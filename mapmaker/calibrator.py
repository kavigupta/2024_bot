import numpy as np
import tqdm

from permacache import permacache

from .constants import TARGET_PV_SPREAD_50
from .data import ec
from .utils import hash_model


def bias(preds):
    dems = preds > 0
    gop = preds < 0
    ecs = np.array(ec())[:, 0]
    return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()


def calibrate(model, *, for_year):
    model = model.with_alpha(1)
    model = model.with_dekurt(0.5)
    for _ in range(10):
        print("OPTIMIZING DEKURT")
        model, state_preds, changed_1 = search(
            model,
            for_year,
            low=0,
            high=model.dekurt_param * 2,
            set_attribute=model.with_dekurt,
            prop=lambda pv: (((pv - np.mean(pv)) / np.std(pv)) ** 4).mean(),
            target=3,
            sensitivity=0.1,
        )
        print("OPTIMIZING ALPHA")
        model, state_preds, changed_2 = search(
            model,
            for_year,
            low=0,
            high=model.alpha * 2,
            set_attribute=model.with_alpha,
            prop=lambda pv: np.percentile(pv, 75) - np.percentile(pv, 25),
            target=TARGET_PV_SPREAD_50,
            sensitivity=0.5e-2,
        )
        if not (changed_1 or changed_2):
            break

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


@permacache(
    "2024bot/calibrator/search",
    key_function=dict(model=hash_model, set_attribute=lambda x: x.__name__),
)
def search(model, for_year, *, low, high, set_attribute, prop, target, sensitivity):
    changed = False
    while True:
        mid = (low + high) / 2
        model = set_attribute(mid)
        _, state_preds, pv = model.family_of_predictions(year=for_year)
        out = prop(pv)
        print(f"Parameter: {mid:.4f}")
        print(f"Property: {out:.2%}")
        if abs(out - target) < sensitivity:
            break
        if out > target:
            high = mid
        else:
            low = mid
        changed = True
    return model, state_preds, changed
