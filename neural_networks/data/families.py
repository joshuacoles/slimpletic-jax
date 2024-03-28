from dataclasses import dataclass
from typing import Union


@dataclass
class Family:
    key: str
    lagrangian: callable
    k_potential: Union[callable, None]
    embedding_shape: tuple[int]


dho = Family(
    'dho',
    lambda q, v, _, embedding: embedding[0] * (v[0] ** 2) - embedding[1] * (q[0] ** 2),
    lambda qp, qm, vp, vm, t, embedding: embedding[2] * vp[0] * qm[0],
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

families = {
    'dho': dho,
    'basic_power_series': basic_power_series,
    'power_series_with_prefactor': power_series_with_prefactor
}


def lookup_family(key: str) -> Family:
    return families[key]
