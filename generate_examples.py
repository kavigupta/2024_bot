import tqdm

from mapmaker.generate_image import get_model, get_image

model = get_model(calibrated=False)
for y in 2012, 2016, 2020:
    model.sample_map(
        f"{y} Actual",
        seed=None,
        path=f"images/{y}_actual.svg",
        year=y,
    )
    model.sample_map(
        f"{y} Pred", seed=None, path=f"images/{y}_pred.svg", year=y, correct=False
    )
    model.sample_map(
        f"{y} Residuals",
        seed=None,
        path=f"images/{y}_residuals.svg",
        year=y,
        correct="just_residuals",
    )
# model.sample_map(
#     "2024 Pred",
#     seed=None,
#     path="images/2024_pred.svg",
#     correct=False,
#     year=2024,
# )
# model.sample_map(
#     "2024 Pred Corrected",
#     seed=None,
#     path="images/2024_pred_corrected.svg",
#     year=2024,
# )
# for i in tqdm.trange(10, 20):
#     get_image(i, i)
