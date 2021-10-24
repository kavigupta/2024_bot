import tqdm

from mapmaker.generate_image import get_model
from mapmaker.mapper import USAPresidencyBaseMap

model = get_model(calibrated=False)
for y in (
    2010,
    2012,
    2014,
    2016,
    2018,
    2020,
):
    model.sample_map(
        f"2020 partisanship, {y} turnout",
        seed=None,
        path=f"turnout_modeling/{y}_actual.svg",
        year=2020,
        turnout_year=y,
        basemap=USAPresidencyBaseMap(),
    )
