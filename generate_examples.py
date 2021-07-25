import tqdm

from mapmaker.generate_image import get_model, get_image

YEARS = (2024, 2022, 2010, 2012, 2014, 2016, 2018, 2020)

MAP_TYPES = ["senate", "president"]

MAP_TYPES = ["senate"]

model = get_model(calibrated=False)
for y in YEARS:
    for map_type in MAP_TYPES:
        if map_type == "senate" and y != 2022:
            continue
        prefix = f"{y}" if map_type == "president" else f"{y}_sen"
        prefix_name = prefix.replace("_sen", " Senate")
        model.sample_map(
            f"{prefix_name} {'Actual' if y <= 2020 else 'Pred Corrected'}",
            seed=None,
            path=f"images/{prefix}_actual.svg",
            year=y,
            map_type=map_type,
        )
        model.sample_map(
            f"{prefix_name} Pred",
            seed=None,
            path=f"images/{prefix}_pred.svg",
            year=y,
            correct=False,
            map_type=map_type,
        )
        model.sample_map(
            f"{prefix_name} Residuals",
            seed=None,
            path=f"images/{prefix}_residuals.svg",
            year=y,
            correct="just_residuals",
            map_type=map_type,
        )
for i in tqdm.trange(10, 20):
    get_image(i, i, map_type="president")
