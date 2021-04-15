import attr

import numpy as np
from sklearn.decomposition import PCA

METADATA = {"FIPS", "biden_2020", "total_votes", "state"}


@attr.s
class Features:
    featurizer = attr.ib()
    metadata_2020 = attr.ib()
    features_2020 = attr.ib()
    features_2024 = attr.ib()

    @staticmethod
    def fit(data_2020, data_2024, pca=20):
        metadata_2020, data_2020 = strip_columns(data_2020)
        _, data_2024 = strip_columns(data_2024)
        featurizer = PCA(pca, whiten=True).fit(data_2020)
        features_2020 = add_ones(featurizer.transform(data_2020))
        features_2024 = add_ones(featurizer.transform(data_2024))
        return Features(featurizer, metadata_2020, features_2020, features_2024)

    def features(self, year):
        return {2020: self.features_2020, 2024: self.features_2024}[year]


def strip_columns(data):
    features = data.fillna(0).copy()
    features = features[[x for x in features if x not in METADATA]]
    return data[list(METADATA)], np.array(features)


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
