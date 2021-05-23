from permacache import stable_hash

import numpy as np


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
