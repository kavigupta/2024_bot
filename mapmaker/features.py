import attr

import numpy as np
from sklearn.decomposition import PCA

METADATA = {"FIPS", "biden_2020", "total_votes", "state", "CVAP", "turnout"}


@attr.s
class Features:
    featurizer = attr.ib()
    features_by_key = attr.ib()

    @staticmethod
    def fit(data_by_key, train_key, *, dimensions):
        data_by_key = {k: strip_columns(v) for k, v in data_by_key.items()}
        featurizer = PCA(dimensions, whiten=True).fit(data_by_key[train_key])
        return Features(
            featurizer,
            {k: add_ones(featurizer.transform(v)) for k, v in data_by_key.items()},
        )

    def features(self, year):
        return self.features_by_key[year]


def metadata(data_by_key, train_key):
    return data_by_key[train_key][sorted(METADATA)]


def strip_columns(data):
    features = data.fillna(0).copy()
    features = features[[x for x in features if x not in METADATA]]
    return np.array(features)


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
