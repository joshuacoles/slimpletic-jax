import jax
import jaxopt
from jax import numpy as jnp


def o_system(x, additional_arg):
    a = additional_arg * x[0]
    b = x[1]

    return jnp.array([
        10 * (6 * b ** 4 - a ** 2),
        1 + 2 * a
    ])


def form_system(
        r: int
):
    def system(
            qi_n,
            q_n,
            pi_n,
            t_n,
            dt,
            r
    ):
        a = qi_n[0]
        b = qi_n[1]

        return jnp.array([
            10 * (6 * b ** 4 - a ** 2),
            1 + 2 * a
        ])

    return system


system = form_system(
    r=2
)

gn = jaxopt.GaussNewton(residual_fun=system)
gn_sol = gn.run(
    jnp.array([1.0, 2.0])
).params
jax.debug.print(
    "sol = {} with f(sol) = {}",
    gn_sol,
    system(gn_sol)
)
