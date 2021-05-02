from permacache import stable_hash


def hash_model(m):
    return stable_hash(
        dict(
            other_dict={k: v for k, v in m.__dict__.items() if not k.startswith("_")},
            state_dict=m.state_dict(),
        )
    )
