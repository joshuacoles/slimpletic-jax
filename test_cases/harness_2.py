import matplotlib.pyplot as plt

from harness import original, solver, lagrangian_f, L, r, dt, dof, q0, pi0, t0, t_sample_count, t
from jax import numpy as jnp
import numpy as np

from original import GalerkinGaussLobatto
from slimpletic import Solver


class Hybrid(Solver):
    sympy_solver: GalerkinGaussLobatto

    def __setattr__(self, key, value):
        if key == 'sympy_solver':
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def compute_qi_values(self, previous_q, previous_pi, t_value):
        jax_cal = super().compute_qi_values(previous_q, previous_pi, t_value)
        values = self.sympy_solver.compute_qi_values(previous_q, previous_pi, t_value, self.dt)
        return jnp.array(values)

    def integrate(self, *args, **kwargs):
        print("WARNING: Using manual integration as the hybrid method does not support JAX JIT.")
        return self.integrate_manual(*args, **kwargs)

hybrid = Hybrid(
    r=r,
    dt=dt,
    lagrangian=lagrangian_f
)

hybrid.sympy_solver = original

print("Testing compute_qi_values **hybrid**")
for _ in range(100):
    previous_q = jnp.array(np.random.rand(dof))
    previous_pi = jnp.array(np.random.rand(dof))
    t = np.random.rand()

    original_values = original.compute_qi_values(previous_q, previous_pi, t, dt)
    jax_values = hybrid.compute_qi_values(previous_q, previous_pi, t)

    try:
        assert jnp.allclose(
            original_values,
            jax_values
        )
    except AssertionError:
        print(original_values)
        print(jax_values)
        print(original_values - jax_values)
        raise

print("Success")

jax_results = hybrid.integrate_manual(jnp.array(q0), jnp.array(pi0), t0, t_sample_count)
original_results = original.integrate(
    q0=np.array(q0),
    pi0=np.array(pi0),
    t=t,
    dt=dt,
)

plt.plot(t, jax_results[0], label='JAX')
plt.plot(t, original_results[0], label='Original')
plt.title('q')
plt.show()

plt.plot(t, jax_results[1], label='JAX')
plt.plot(t, original_results[1], label='Original')
plt.title('pi')
plt.show()
