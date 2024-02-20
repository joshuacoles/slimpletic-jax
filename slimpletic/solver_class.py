from functools import partial
from typing import Any, Callable, Union

import jax

import jax.numpy as jnp
import jaxopt
from attr.exceptions import FrozenError

from slimpletic.ggl import ggl, dereduce
from slimpletic.helpers import fill_out_initial


# Helper for default values of lagrangian and k_potential
def zero_function(*args):
    return 0


class Solver:
    r: int
    dt: float

    optimiser: jaxopt.GaussNewton
    lagrangian: callable
    derivatives: callable

    k_potential: callable
    k_derivatives: callable

    xs: jnp.ndarray
    ws: jnp.ndarray
    dij: jnp.ndarray

    # This is a bit of a hack to ensure that the object is immutable after creation.
    _sealed: bool = False

    def __setattr__(self, key, value):
        """
        When running under JAX JIT, our methods will not be recomputed if the object is mutated. Hence, to avoid bugs we
        seal the object after creation.

        With the new ability to change the lagrangian and k_potential, we need to unseal the object when we change these
        parameters. This is done by checking if the key is '_sealed', and if it is we allow the change to go through.
        """
        if self._sealed and key != '_sealed':
            raise FrozenError("You cannot edit the solver after it has been created. Use create_similar to create a "
                              "new instance with the same parameters.")
        else:
            super().__setattr__(key, value)

    def create_similar(self, **kwargs):
        """
        This is a helper function to create a new instance of the solver with the same parameters as the current one.

        :param kwargs: The parameters to override in the new instance.
        :return: A new instance which inherits the parameters of the current instance, with the overridden parameters.
        """
        return Solver(
            r=kwargs.get('r') or self.r,
            dt=kwargs.get('dt') or self.dt,
            lagrangian=kwargs.get('lagrangian') or self.lagrangian,
            k_potential=kwargs.get('k_potential') or self.k_potential,
        )

    def __init__(
            self,
            r: int,
            dt: float,
            lagrangian: Union[None, Callable[[Any, Any, Any], Any]] = None,
            k_potential: Union[None, Callable[[Any, Any, Any, Any, Any], Any]] = None,
            jit_residue: bool = True
    ):
        self.r = r
        self.xs, self.ws, self.dij = dereduce(ggl(r), dt)
        self.dt = dt

        # self.lagrangian = lagrangian or zero_function
        # self.k_potential = k_potential or zero_function

        self.set_lagrangian(
            lagrangian=lagrangian,
            k_potential=k_potential,
            jit_residue=jit_residue
        )

        self._sealed = True

    def set_lagrangian(
            self,
            lagrangian: Union[None, Callable[[Any, Any, Any], Any]] = None,
            k_potential: Union[None, Callable[[Any, Any, Any, Any, Any], Any]] = None,
            jit_residue: bool = True
    ):
        self._sealed = False
        self.lagrangian = lagrangian or zero_function
        self.k_potential = k_potential or zero_function

        if jit_residue:
            self._optimiser = jaxopt.GaussNewton(residual_fun=jax.jit(self.residue))
        else:
            self._optimiser = jaxopt.GaussNewton(residual_fun=self.residue)

        self._sealed = True

    def derivatives(self, qi_values, t0):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.lagrangian_d, argnums=0)(qi_values, t0)

    def k_derivatives(self, qi_plus_values, qi_minus_values, t0):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.k_potential_d, argnums=(0, 1))(qi_plus_values, qi_minus_values, t0)

    def lagrangian_d(self, qi_values, t0):
        # Eq. 4 (part 2)
        t_quadrature_values = t0 + (1 + self.xs) * self.dt / 2

        # Eq. 6. Given the values of qi we can compute the values of qidot at the quadrature points.
        qidot_vec = jax.numpy.matmul(self.dij, qi_values)

        # Eq. 7, first evaluate the function at the quadrature points, then compute the weighted sum.
        fn_i = jax.vmap(self.lagrangian)(qi_values, qidot_vec, t_quadrature_values)
        return jnp.dot(self.ws, fn_i)

    def k_potential_d(
            self,
            qi_plus_values,
            qi_minus_values,
            t0
    ):
        # Eq. 4 (part 2)
        t_quadrature_values = t0 + (1 + self.xs) * self.dt / 2

        # Eq. 6. Given the values of qi we can compute the values of qidot at the quadrature points.
        qi_plus_dot_vec = jax.numpy.matmul(self.dij, qi_plus_values)
        qi_minus_dot_vec = jax.numpy.matmul(self.dij, qi_minus_values)

        # Eq. 7, first evaluate the function at the quadrature points, then compute the weighted sum.
        fn_i = jax.vmap(self.k_potential)(qi_plus_values, qi_minus_values, qi_plus_dot_vec, qi_minus_dot_vec,
                                          t_quadrature_values)
        return jnp.dot(self.ws, fn_i)

    def compute_qi_values(self, previous_q, previous_pi, t_value):
        optimiser_result = self._optimiser.run(
            fill_out_initial(previous_q, r=self.r - 1),
            t_value,
            previous_q,
            previous_pi
        )

        return jnp.insert(optimiser_result.params, 0, previous_q, axis=0)

    def compute_pi_next(self, qi_values, t_value):
        # Eq 13(b)
        dld_dqi_values = self.derivatives(qi_values, t_value)
        dkd_dqi_plus_values, dkd_dqi_minus_values = self.k_derivatives(qi_values,
                                                                       jnp.zeros_like(qi_values, dtype=float), t_value)
        return dld_dqi_values[-1] + dkd_dqi_minus_values[-1]

    def residue(self, trailing_qi_values, t, q0, pi0):
        """
        Compute the residue for the optimiser based on Equations 13(a) and 13(c) in the paper.

        NOTE: The trailing_qi_values array does not and cannot include the initial value of q at t0 as this is
        necessarily fixed, and not subject to optimisation. Hence, when passing qi_values around we need to insert the
        initial value of q back into the array.

        :param trailing_qi_values: The values of q at the quadrature points, **after** the initial point.
        :param t: The time at which we are computing the residue.
        :param q0: The initial value of q.
        :param pi0: The initial value of pi.
        :return: The residue for the optimiser, this will be zero when the solution is found.
        """
        qi_values = jnp.insert(trailing_qi_values, 0, q0, axis=0)
        dld_dqi_values = self.derivatives(qi_values, t)

        # Evaluate in the physical limit
        dkd_dqi_plus_values, dkd_dqi_minus_values = self.k_derivatives(
            qi_values,
            jnp.zeros_like(qi_values, dtype=float),
            t
        )

        # Eq 13(a), we set the derivative wrt to the initial point to negative of pi0
        eq13a_residue = pi0 + dld_dqi_values[0] + dkd_dqi_minus_values[0]

        # Eq 13(c), we set the derivative wrt to each interior point to zero
        eq13c_residues = dld_dqi_values[1:-1] + dkd_dqi_minus_values[1:-1]

        return jnp.append(
            eq13c_residues,
            eq13a_residue
        )

    def compute_next(
            self,
            previous_state,
            t_value
    ):
        (previous_q, previous_pi) = previous_state
        qi_values = self.compute_qi_values(previous_q, previous_pi, t_value)

        # q_{n, r + 1} = q_{n + 1, 0}
        q_next = qi_values[-1]
        pi_next = self.compute_pi_next(qi_values, t_value)
        next_state = (q_next, pi_next)

        return next_state, next_state

    @partial(jax.jit, static_argnums=(0, 4))
    def integrate(self, q0: jnp.ndarray, pi0: jnp.ndarray, t0: float, iterations: int,
                  result_orientation: str = 'time'):
        """
        Integrate the system using the slimpletic method.

        :param q0: The initial value of q, expected to be a jnp array of shape (dof,)
        :param pi0: The initial value of pi, expected to be a jnp array of shape (dof,)
        :param t0: The initial value of t
        :param iterations: How many iterations of size dt to perform when integrating the system.
        :param result_orientation: If results should be returned such that array[i] is the value of the i-th coordinate
        along the time axis, or if array[i] is the value of the coordinates at the i-th time step.
        :return: The values of q and pi along the evolution of the system, oriented as specified by `result_orientation`.
        """
        if not (isinstance(q0, jnp.ndarray) and isinstance(pi0, jnp.ndarray)):
            raise ValueError("q0 and pi0 must be jax numpy arrays.")

        if q0.shape != pi0.shape:
            raise ValueError("q0 and pi0 must have the same shape.")

        if result_orientation not in ['time', 'coordinate']:
            raise ValueError("orientation must be either 'time' or 'coordinate'.")

        # These are the values of t which we will sample the solution at. This does not include the initial value of t
        # as the initial state of the system is already known.
        t_samples = t0 + (1 + jnp.arange(iterations)) * self.dt

        _, (q, pi) = jax.lax.scan(
            f=self.compute_next,
            xs=t_samples,
            init=(q0, pi0),
        )

        # We need to add the initial values back into the results.
        q_with_initial = jnp.insert(q, 0, q0, axis=0)
        pi_with_initial = jnp.insert(pi, 0, pi0, axis=0)

        if result_orientation == 'time':
            return q_with_initial, pi_with_initial
        else:
            return q_with_initial.T, pi_with_initial.T

    def _integrate_manual(self, q0, pi0, t0, iterations):
        """
        This is a manual implementation of the integrate function, which is useful for debugging and understanding the
        code. It is not recommended for usage use as it is *much* slower than the standard integrate function, and
        does not implement all the same features.
        """
        import sys
        print("Warning: Using manual integration, this is much slower than the standard integration function with fewer"
              " features. This should only be used for debugging.", file=sys.stderr)

        if not (isinstance(q0, jnp.ndarray) and isinstance(pi0, jnp.ndarray)):
            raise ValueError("q0 and pi0 must be jax numpy arrays.")

        # These are the values of t which we will sample the solution at. This does not include the initial value of t
        # as the initial state of the system is already known.
        t_samples = t0 + (1 + jnp.arange(iterations)) * self.dt

        q = [q0]
        pi = [pi0]

        carry = (q0, pi0)

        for t in t_samples:
            carry, (q_next, pi_next) = self.compute_next(carry, t)
            q.append(q_next)
            pi.append(pi_next)

        return jnp.stack(q), jnp.stack(pi)
