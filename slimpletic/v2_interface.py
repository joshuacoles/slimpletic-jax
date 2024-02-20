import time
from functools import partial
from typing import Union, Callable

import jax.lax
import jax.numpy
import jaxopt
import numpy as np
from jax import numpy as jnp
from timeit import Timer

from slimpletic.ggl import ggl, dereduce
from slimpletic.helpers import fill_out_initial
from slimpletic.solver_class import zero_function


class GGLBundle:
    """
    A bundle of the values of the GGL quadrature method, which are used to compute the values of the quadrature points.
    
    The computation of these is kept separate from the DiscretisedSystem class as they are not expected to change, and
    are less amenable to JIT compilation and gradient computation.
    """

    def __init__(self, r: int):
        self.r = r
        self.xs, self.ws, self.dij = ggl(r)


class DiscretisedSystem:
    r: int
    dt: float

    xs: jnp.ndarray
    ws: jnp.ndarray
    dij: jnp.ndarray

    lagrangian: Callable
    k_potential: Callable

    # The number of iterations to batch together when integrating the system. If False, no batching is performed.
    batch_size: Union[int, None] = 100

    def __init__(
            self,
            ggl_bundle: GGLBundle,
            dt: float,
            lagrangian: Union[None, Callable] = None,
            k_potential: Union[None, Callable] = None,
    ):
        self.r = ggl_bundle.r
        self.xs, self.ws, self.dij = dereduce((ggl_bundle.xs, ggl_bundle.ws, ggl_bundle.dij), dt)
        self.dt = dt
        self.lagrangian = lagrangian or zero_function
        self.k_potential = k_potential or zero_function
        self._optimiser = jaxopt.GaussNewton(residual_fun=self.residue)

    def lagrangian_d(self, qi_values, t0):
        print("LAG_D")
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
        fn_i = jax.vmap(self.k_potential)(
            qi_plus_values,
            qi_minus_values,
            qi_plus_dot_vec,
            qi_minus_dot_vec,
            t_quadrature_values
        )

        return jnp.dot(self.ws, fn_i)

    def derivatives(self, qi_values, t0):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.lagrangian_d, argnums=0)(qi_values, t0)

    def k_derivatives(self, qi_plus_values, qi_minus_values, t0):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.k_potential_d, argnums=(0, 1))(qi_plus_values, qi_minus_values, t0)

    def compute_qi_values(self, previous_q, previous_pi, t_value):
        print("COMPUTE_QI_VALUES")
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
                                                                       jnp.zeros_like(qi_values, dtype=float),
                                                                       t_value)
        return dld_dqi_values[-1] + dkd_dqi_minus_values[-1]

    @partial(jax.jit, static_argnums=(0,))
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
        print("RESIDUE")
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

    @partial(jax.jit, static_argnums=(0,))
    def compute_next(
            self,
            previous_state,
            t_value
    ):
        print("COMPUTE_NEXT")
        (previous_q, previous_pi) = previous_state
        qi_values = self.compute_qi_values(previous_q, previous_pi, t_value)

        # q_{n, r + 1} = q_{n + 1, 0}
        q_next = qi_values[-1]
        pi_next = self.compute_pi_next(qi_values, t_value)
        next_state = (q_next, pi_next)

        return next_state, next_state

    @partial(jax.jit, static_argnums=(0,))
    def _integrate_inner_batch(
            self,
            carry: tuple[jnp.ndarray, jnp.ndarray],
            ts: jnp.ndarray,
    ):
        (q0, pi0) = carry

        _, (q, pi) = jax.lax.scan(
            f=self.compute_next,
            xs=ts,
            length=self.batch_size,
            init=(q0, pi0),
        )

        return (q[-1], pi[-1]), (q, pi)

    def integrate(
            self,
            q0: jnp.ndarray,
            pi0: jnp.ndarray,
            t0: float,
            iterations: int,
            result_orientation: str = 'time'
    ):
        """
        Integrate the system using the slimpletic method.

        :param q0: The initial value of q, expected to be a jnp array of shape (dof,)
        :param pi0: The initial value of pi, expected to be a jnp array of shape (dof,)
        :param t0: The initial value of t
        :param iterations: How many iterations of size dt to perform when integrating the 
        :param result_orientation: If results should be returned such that array[i] is the value of the i-th coordinate
        along the time axis, or if array[i] is the value of the coordinates at the i-th time step.
        :return: The values of q and pi along the evolution of the system, oriented as specified by `result_orientation`.
        """
        print(f"INTEGRATE {time.time_ns()}")

        if not (isinstance(q0, jnp.ndarray) and isinstance(pi0, jnp.ndarray)):
            raise ValueError("q0 and pi0 must be jax numpy arrays.")

        if q0.shape != pi0.shape:
            raise ValueError("q0 and pi0 must have the same shape.")

        if result_orientation not in ['time', 'coordinate']:
            raise ValueError("orientation must be either 'time' or 'coordinate'.")

        if self.batch_size is not None:
            timer_3 = time.time_ns()
            # We use numpy to compute the number of batches, as this is static and should NOT be considered during JIT.
            number_of_batches = np.ceil(iterations / self.batch_size)
            print("Number of batches: ", number_of_batches)
            t_samples_extended = t0 + (1 + jnp.arange(self.batch_size * number_of_batches)) * self.dt
            t_samples_extended = t_samples_extended.reshape(-1, self.batch_size)
            print("Timer 3: ", (time.time_ns() - timer_3) / 10e9)
            timer_4 = time.time_ns()

            qs = []
            pis = []
            q_previous = q0
            pi_previous = pi0

            for i in range(int(number_of_batches)):
                (q_previous, pi_previous), (q, pi) = self._integrate_inner_batch(
                    carry=(q_previous, pi_previous),
                    ts=t_samples_extended[i]
                )

                qs.append(q)
                pis.append(pi)

            q = jnp.concatenate(qs, axis=0)
            pi = jnp.concatenate(pis, axis=0)

            print("Timer 4: ", (time.time_ns() - timer_4) / 10e9)
            timer_5 = time.time_ns()

            q = q[:iterations]
            pi = pi[:iterations]
            print("Timer 5: ", (time.time_ns() - timer_5) / 10e9)
        else:
            # These are the values of t which we will sample the solution at. This does not include the initial value of t
            # as the initial state of the system is already known.
            t_samples = t0 + (1 + jnp.arange(iterations)) * self.dt

            _, (q, pi) = jax.lax.scan(
                f=self.compute_next,
                xs=t_samples,
                init=(q0, pi0),
            )

        # This actually takes a long time to compute compared to everything else after JIT
        timer_6 = time.time_ns()
        # We need to add the initial values back into the results.
        q_with_initial = jnp.insert(q, 0, q0, axis=0)
        pi_with_initial = jnp.insert(pi, 0, pi0, axis=0)
        print("Timer 6: ", (time.time_ns() - timer_6) / 10e9)

        if result_orientation == 'time':
            return q_with_initial, pi_with_initial
        else:
            return q_with_initial.T, pi_with_initial.T

class SolverManual(DiscretisedSystem):
    def integrate_manual(
            self,
            q0,
            pi0,
            t0,
            iterations
    ):
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
