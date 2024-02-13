import numpy as np

from harness import original, solver, lagrangian_f, L, r, dt
from jax import numpy as jnp, config as jax_config
import numpy as np

assert jax_config.read('jax_enable_x64')


# The first thing to check is that the lagrangians are actually the same

def eval_sympy_lagrangian(q: jnp.ndarray, v: jnp.ndarray, t: float):
    return float(L.subs([
        *zip(original.q, q),
        *zip(original.v, v),
        (original.t, t)
    ]))


dof = original.degrees_of_freedom

print("Testing lagrangian")
for _ in range(100):
    q = jnp.array(np.random.rand(dof))
    v = jnp.array(np.random.rand(dof))
    t = np.random.rand()

    assert jnp.isclose(
        original.lagrangian(q, v, t),
        solver.lagrangian(q, v, t)
    )

print("Success")

print("Testing lagrangian_d")
# The next thing is to test the discretised lagrangians are the same
for _ in range(100):
    qi_values = jnp.array(np.random.normal(size=(r + 2, dof)))
    t = np.random.rand()

    assert jnp.isclose(
        original.lagrangian_d(qi_values, t, dt),
        solver.lagrangian_d(qi_values, t)
    )
print("Success")


