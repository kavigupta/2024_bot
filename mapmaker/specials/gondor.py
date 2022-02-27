import copy
from functools import lru_cache
import json
import os

from cached_property import cached_property
import apportionpy.methods.huntington_hill

import geopandas
import numpy as np
import pandas as pd

from shapely.ops import unary_union, transform

from ..model import Model
from ..mapper import BaseMap, draw_ec, draw_tipping_point
from ..colors import Profile

PV_SPREAD = 0.07

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class GondorMap(BaseMap):
    @cached_property
    def load_ec(self):
        df = geopandas.read_file(os.path.join(ROOT, "shapefiles/gondor/Gondor.shp"))
        senate = 2
        _, sh, *_ = apportionpy.methods.huntington_hill.calculate_huntington_hill(
            800 - len(df.Population) * senate, list(df.Population * df.IncomeLog)
        )
        df["electoral_college"] = [x + senate for x in sh]
        return df

    @cached_property
    def load_file(self):
        df = geopandas.read_file(
            os.path.join(ROOT, "shapefiles/gondor/Gondorincome revised.shp")
        )
        df["id"] = df.PrecName
        df["FIPS"] = df.id
        df["CVAP"] = df.Totalpop
        df["state"] = df.County
        df["DunedainNoEd_Noelv_Norelig"] = (
            df["DunedainNo"] * (1 - df["DunElvish"]) * (1 - df["DunNoRelig"])
        )
        df["DunedainNoEd_Noelv_Relig"] = (
            df["DunedainNo"] * (1 - df["DunElvish"]) * df["DunNoRelig"]
        )
        df["DunedainNoEd_Elv_Norelig"] = (
            df["DunedainNo"] * df["DunElvish"] * (1 - df["DunNoRelig"])
        )
        df["DunedainNoEd_Elv_Relig"] = (
            df["DunedainNo"] * df["DunElvish"] * df["DunNoRelig"]
        )
        return fix_bounds(df)

    @property
    def counties(self):
        geojson = json.loads(self.load_file.to_json())
        for f in geojson["features"]:
            f["id"] = str(f["properties"]["id"])
        return geojson

    @property
    def metadata(self):
        df = self.load_file
        return df[["FIPS", "state"]]

    @property
    def electoral_votes(self):
        return self.load_ec[["County", "electoral_college"]].set_index("County")

    @property
    def county_plotly_kwargs(self):
        return dict(marker_line_width=0.3)

    @property
    def state_plotly_kwargs(self):
        return dict(marker_line_width=1.3)

    @property
    def map_scale(self):
        return 0.85

    @property
    def map_dy(self):
        return 0

    def county_mask(self, year):
        return 1

    def draw_topline(self, *args, **kwargs):
        return draw_ec(self, *args, **kwargs)

    def draw_tipping_point(self, *args, **kwargs):
        return draw_tipping_point(self, *args, **kwargs)

    def modify_figure_layout(self, figure, profile):
        figure.update_geos(
            showcoastlines=False,
            showland=False,
            showocean=False,
            showlakes=False,
            bgcolor=profile.background_hex(),
            framewidth=0,
        )

    def extra_county_maps(self, profile):
        return [
            self.create_states_outline(
                profile, marker_line_color="white", marker_line_width=0.5
            )
        ]

    @property
    def insets(self):
        from types import SimpleNamespace

        y_top = 330
        x_left = 25
        scale = 0.23
        return {
            "west": SimpleNamespace(
                name="Dol Amroth",
                scale=scale,
                x_out=x_left,
                y_out=y_top,
                x_in=[12, 25],
                y_in=[-25, -10],
                text_dx=40,
            ),
            "east-nested-inset": SimpleNamespace(
                name="Minas Tirith",
                scale=scale,
                x_out=x_left + 90,
                y_out=y_top,
                x_in=[135.5, 136.8],
                y_in=[22.8, 24.3],
                text_dx=28,
            ),
            "east": SimpleNamespace(
                name="Osgiliath-Pelennor",
                scale=scale,
                x_out=x_left + 200,
                y_out=y_top,
                x_in=[137, 146],
                y_in=[23, 30],
                text_dx=8,
            ),
        }


class GondorDemographicModel:
    def __init__(self):
        self.partisanship_2020 = {
            "DunedainEd": 0.04,
            "DunedainNoEd_Noelv_Norelig": 0.25,
            "DunedainNoEd_Noelv_Relig": 0.01,
            "DunedainNoEd_Elv_Norelig": -0.3,
            "DunedainNoEd_Elv_Relig": -0.4,
            "Druedain": 0.8,
            "Eotheod": -0.25,
            "Southron": -0.1,
        }
        self.turnout_2020 = {
            "DunedainEd": 0.8,
            "DunedainNoEd_Noelv_Norelig": 0.5,
            "DunedainNoEd_Noelv_Relig": 0.61,
            "DunedainNoEd_Elv_Norelig": 0.6,
            "DunedainNoEd_Elv_Relig": 0.7,
            "Druedain": 0.55,
            "Eotheod": 0.75,
            "Southron": 0.35,
        }
        self.partisan_income_coefficient = -0.3
        self.turnout_income_coefficient = 0
        self.noise_seed = 0

    @property
    def demos(self):
        return sorted(self.partisanship_2020)

    def perturb(self, prediction_seed, alpha_partisanship, alpha_turnout):
        if prediction_seed is None:
            return self
        perturbed = copy.deepcopy(self)
        pseed, tseed, incomeseed, nseed = np.random.RandomState(prediction_seed).choice(
            2 ** 32, size=4
        )
        perturbed.partisanship_2020 = self._perturb_dict(
            self.partisanship_2020, alpha_partisanship, pseed, -1, 1
        )
        perturbed.turnout_2020 = self._perturb_dict(
            self.turnout_2020, alpha_turnout, tseed, 0, 1
        )
        cpi, cti = np.random.RandomState(incomeseed).randn(2) * 0.3 * alpha_partisanship
        perturbed.partisan_income_coefficient += cpi
        perturbed.turnout_income_coefficient += cti * 0
        perturbed.noise_seed = nseed
        return perturbed

    def _perturb_dict(self, d, alpha, seed, min_val, max_val):
        values = np.array([d[demo] for demo in self.demos])
        value_to_add = np.random.RandomState(seed).randn(*values.shape) * alpha
        values = self._add_in_atan_space(values, value_to_add, min_val, max_val)
        return {demo: val for demo, val in zip(self.demos, values)}

    def _add_in_atan_space(self, values, value_to_add, min_val, max_val):
        # scale (min_val, max_val) to (-1, 1)
        values = values - min_val
        # now in range (0, max_val - min_val)
        values = values / (max_val - min_val) * 2
        # now in range (0, 2)
        values = values - 1
        # now in range (-1, 1)
        values = np.arctanh(values)
        # now in range (-inf, inf)
        values = values + value_to_add
        # now in range (-inf, inf)
        values = np.tanh(values)
        # now in range (-1, 1)
        values += 1
        # now in range (0, 2)
        values = values / 2 * (max_val - min_val)
        # now in range (0, max_val - min_val)
        values += min_val
        return values

    def predict(self, data, correct):
        demo_values = np.array(data[self.demos])
        nonzero = demo_values.sum(1) != 0
        demo_values[nonzero] += np.random.RandomState(self.noise_seed).choice(
            10, replace=False
        )
        demo_values = demo_values / np.maximum(demo_values.sum(1)[:, None], 1)
        p = np.array([self.partisanship_2020[demo] for demo in self.demos])
        t = np.array([self.turnout_2020[demo] for demo in self.demos])

        income_normalized = np.array(np.log(data.Incometest + 1))
        income_normalized = income_normalized - income_normalized.mean()
        income_normalized = income_normalized / income_normalized.std()

        p = self._add_in_atan_space(
            p[None],
            self.partisan_income_coefficient * income_normalized[:, None],
            -1,
            1,
        )
        t = self._add_in_atan_space(
            t[None], self.turnout_income_coefficient * income_normalized[:, None], 0, 1
        )
        pt = p * t

        pt = (demo_values * pt).sum(1)
        t = (demo_values * t).sum(1)

        p = np.divide(pt, t, out=np.zeros_like(pt), where=np.abs(t) > 1e-5)

        assert t.max() <= 1
        assert t.min() >= 0
        assert p.max() <= 1
        assert p.min() >= -1
        return p, t


class GondorModel(Model):
    def __init__(self):
        super().__init__(
            data_by_year={2020: GondorMap().load_file},
            feature_kwargs=dict(dimensions=None),
        )
        self.model = GondorDemographicModel()

    def fully_random_sample(
        self, *, year, prediction_seed, correct, turnout_year, basemap
    ):
        assert year == 2020
        assert turnout_year == None
        model = self.model.perturb(
            prediction_seed=prediction_seed,
            alpha_partisanship=self.alpha,
            alpha_turnout=self.alpha * 10,
        )
        return model.predict(
            data=self.data[year],
            correct=correct,
        )

    def get_demographics_by_county(self, *, year):
        assert not "implemented"


def fix_bounds(d):
    u = unary_union([x.buffer(0) for x in d.geometry])
    mi_x, mi_y, ma_x, ma_y = u.bounds

    t_mi_x, t_mi_y, t_ma_x, t_ma_y = -170, -90, 170, 90
    mi_x, ma_x, mi_y, ma_y
    stretch = min((t_ma_y - t_mi_y) / (ma_y - mi_y), (t_ma_x - t_mi_x) / (ma_x - mi_x))

    def deform(xs, ys):
        xs, ys = np.array(xs), np.array(ys)
        xs -= (mi_x + ma_x) / 2
        ys -= (mi_y + ma_y) / 2
        xs *= stretch
        ys *= stretch
        xs += (t_mi_x + t_ma_x) / 2
        ys += (t_mi_y + t_ma_y) / 2
        return xs, ys

    d.geometry = d.geometry.apply(lambda x: transform(deform, x))
    return d


@lru_cache(None)
def get_gondor_model(is_calibrated):
    from ..calibrator import calibrate

    gondor = GondorMap()
    model = GondorModel()
    if is_calibrated:
        model = calibrate(gondor, model, for_year=2020, pv_spread=PV_SPREAD)
    return model


def generate_gondor_map(seed, title, path):

    model = get_gondor_model(seed is not None)

    dem = [
        [0, "#151553"],
        [0.125, "#17176c"],
        [0.25, "#1c1c8c"],
        [0.375, "#2828ac"],
        [0.5, "#3333cc"],
        [0.625, "#4747eb"],
        [0.75, "#6666fb"],
        [0.875, "#aaaeff"],
        [0.9375, "#d5d9ff"],
        [0.9875, "#c0c0c0"],
    ]
    gop = [[k, v[0] + v[5:7] + v[3:5] + v[1:3]] for k, v in dem]

    gondor_profile = Profile(
        symbol=dict(dem="D", gop="R"),
        hue=dict(dem=2 / 3, gop=1),
        bot_name="ElectionsGondor",
        credit="by @Thorongil16 and @notkavi, based on @bot_2024 engine",
        county_colorscales=dict(dem=dem, gop=gop),
        compute_state_via_county=dict(safe=0.4, likely=0.3, lean=0.2, tilt=0.1),
        background_color=(60, 60, 60),
    )

    return model.sample_map(
        title=title,
        path=path,
        year=2020,
        basemap=GondorMap(),
        seed=seed,
        profile=gondor_profile,
        full_output=True,
    )
