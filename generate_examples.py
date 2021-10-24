import tqdm

from mapmaker.generate_image import get_model, get_image
from mapmaker.mapper import USAPresidencyBaseMap, USASenateBaseMap

YEARS = (2024, 2022, 2010, 2012, 2014, 2016, 2018, 2020)

MAP_TYPES = [USAPresidencyBaseMap(), USASenateBaseMap()]

model = get_model(calibrated=False)
for y in YEARS:
    for basemap in MAP_TYPES:
        if isinstance(basemap, USASenateBaseMap) and y != 2022:
            continue
        prefix = f"{y}" if isinstance(basemap, USAPresidencyBaseMap) else f"{y}_sen"
        prefix_name = prefix.replace("_sen", " Senate")
        model.sample_map(
            f"{prefix_name} {'Actual' if y <= 2020 else 'Pred Corrected'}",
            seed=None,
            path=f"images/{prefix}_actual.svg",
            year=y,
            basemap=basemap,
        )
        model.sample_map(
            f"{prefix_name} Pred",
            seed=None,
            path=f"images/{prefix}_pred.svg",
            year=y,
            correct=False,
            basemap=basemap,
        )
        model.sample_map(
            f"{prefix_name} Residuals",
            seed=None,
            path=f"images/{prefix}_residuals.svg",
            year=y,
            correct="just_residuals",
            basemap=basemap,
        )
for i in tqdm.trange(10, 20):
    get_image(i, i, basemap=USAPresidencyBaseMap())
