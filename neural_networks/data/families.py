from dataclasses import dataclass
from typing import Union

import jax


@dataclass
class Family:
    key: str
    lagrangian: callable
    k_potential: Union[callable, None]
    embedding_shape: tuple[int]

    @staticmethod
    def from_key(key: str):
        return lookup_family(key)

    def get_config(self):
        return {
            'key': self.key,
        }

    @staticmethod
    def from_config(config):
        return lookup_family(config['key'])


dho = Family(
    'dho',
    lambda q, v, _, embedding: embedding[0] * (v[0] ** 2) - embedding[1] * (q[0] ** 2),
    lambda qp, qm, vp, vm, t, embedding: -embedding[2] * vp[0] * qm[0],
    (3,)
)

basic_power_series = Family(
    'basic_power_series',
    lambda q, v, _, embedding: embedding[0] * (q[0] ** 2) +
                               embedding[1] * (v[0] ** 2) +
                               embedding[2] * (q[0] * v[0]),
    None,
    (3,)
)

power_series_with_prefactor = Family(
    'power_series_with_prefactor',
    lambda q, v, _, embedding: embedding[3] * (embedding[0] * (q[0] ** 2) +
                                               embedding[1] * (v[0] ** 2) +
                                               embedding[2] * (q[0] * v[0])),
    None,
    (4,)
)

aengus_original = Family(
    'aengus_original',
    lambda q, v, _, an: jax.lax.fori_loop(0, 2,
                                          lambda i, acc: acc + (an[2 * i] * q[0] ** (i + 1)) + (
                                                  an[2 * i + 1] * v[0] ** (i + 1)),
                                          0.0),
    None,
    (4,)
)

families = {
    'dho': dho,
    'basic_power_series': basic_power_series,
    'power_series_with_prefactor': power_series_with_prefactor,
    'aengus_original': aengus_original,
}


def lookup_family(key: str) -> Family:
    try:
        return families[key]
    except KeyError:
        raise KeyError(f"Family {key} not found")
