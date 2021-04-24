import numpy as np
import tqdm

from .data import ec


def bias(preds):
    dems = preds > 0
    gop = preds < 0
    ecs = np.array(ec())[:, 0]
    return ((ecs * dems).sum(1) > (ecs * gop).sum(1)).mean()


def unbias_predictor(model, *, for_year, target_pv_spread_90=17.5e-2):
    alphas = np.arange(0.05, 0.25, 0.05)
    models = [model.with_alpha(alpha) for alpha in alphas]
    preds_each = [
        model.family_of_predictions(year=for_year) for model in tqdm.tqdm(models)
    ]
    pv_spread_90 = np.array(
        [np.percentile(pv, 95) - np.percentile(pv, 5) for _, pv in preds_each]
    )
    print("Alpha values:", alphas)
    print("Spreads:", pv_spread_90 * 100)
    idx = np.argmin(np.abs(pv_spread_90 - target_pv_spread_90))
    print("Optimal alpha:", alphas[idx])
    print("Optimal spread:", pv_spread_90[idx])
    model = models[idx]
    state_preds, _ = preds_each[idx]

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
    model = model.with_predictor(model.predictor.with_bias(best_bias))
    return model
