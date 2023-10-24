import jax
import jaxopt
from jax import numpy as jnp

from discretise import take_vector_grad


def o_system(x, additional_arg):
    a = additional_arg * x[0]
    b = x[1]

    return jnp.array([
        10 * (6 * b ** 4 - a ** 2),
        1 + 2 * a
    ])


def form_system(
        r: int,
        ld,
        dof: int = 2,
):
    def system(
            qi_n,
            q_n,
            pi_n,
            t_n,
            dt,
    ):
        eom_13a = [
            pi_n[d] + differentiate(ld, wrt=['q', d])(qi_n, q_n, pi_n, t_n, dt)
            for d in range(dof)
        ]

        eom_13c = [
            pi_n[d] + differentiate(ld, wrt=['qi', d, i])(qi_n, q_n, pi_n, t_n, dt)
            # Interior points
            for i in range(1, r + 1)
            for d in range(dof)
        ]

        return jnp.array([
            *eom_13a,
            *eom_13c
        ])

    return system


system = form_system(
    r=2
)

gn = jaxopt.GaussNewton(residual_fun=system)
qi_soln = gn.run(
    jnp.array([1.0, 2.0]),
    pi_0 = ...,

).params
jax.debug.print(
    "sol = {} with f(sol) = {}",
    qi_soln,
    system(qi_soln)
)
