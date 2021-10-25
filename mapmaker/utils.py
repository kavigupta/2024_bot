from collections import defaultdict
from permacache import stable_hash

import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def hash_model(m):
    return stable_hash(
        dict(
            other_dict={k: v for k, v in m.__dict__.items() if not k.startswith("_")},
            state_dict=m.state_dict(),
        )
    )


def intersect_all(iterable):
    iterable = list(iterable)
    result = iterable[0]
    for x in iterable[1:]:
        result = np.intersect1d(result, x)
    return result


def counties_to_states(data, counties_geojson):
    shapes = {f["id"]: shape(f["geometry"]) for f in counties_geojson["features"]}
    states = defaultdict(list)
    for fips, state in zip(data["FIPS"], data["state"]):
        states[state].append(shapes[str(fips)])
    states = {k: unary_union(v) for k, v in states.items()}
    return dict(
        type=counties_geojson["type"],
        features=[
            dict(type="Feature", geometry=mapping(sh), id=ident)
            for ident, sh in states.items()
        ],
    )
