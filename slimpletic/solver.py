import time
from functools import partial
from typing import Union, Callable, Any

import jax.lax
import jax.numpy
import jaxopt
import numpy as np
from jax import numpy as jnp

from .ggl import GGLBundle, dereduce
from .helpers import fill_out_initial, zero_function

class DiscretisedSystem:
    # This is Pytree-able
    r: int
    dt: float

    # This is Pytree-able
    xs: jnp.ndarray
    ws: jnp.ndarray
    dij: jnp.ndarray

    # Okay these can't be put into a Pytree
    _lagrangian: Callable
    _k_potential: Callable

    # This is necessarily static
    pass_additional_data: bool
    _optimiser: jaxopt.GaussNewton

    def __init__(
            self,
            ggl_bundle: GGLBundle,
            dt: float,
            lagrangian: Union[None, Callable] = None,
            k_potential: Union[None, Callable] = None,
            pass_additional_data: bool = False
    ):
        self.r = ggl_bundle.r
        self.xs, self.ws, self.dij = dereduce((ggl_bundle.xs, ggl_bundle.ws, ggl_bundle.dij), dt)
        self.dt = dt
        self._lagrangian = lagrangian or zero_function
        self._k_potential = k_potential or zero_function
        self.pass_additional_data = pass_additional_data
        self._optimiser = jaxopt.GaussNewton(residual_fun=self.residue)

    def lagrangian(self, q, qdot, t, additional_data):
        if self.pass_additional_data:
            return self._lagrangian(q, qdot, t, additional_data)
        else:
            return self._lagrangian(q, qdot, t)

    def k_potential(self, q_plus, q_minus, qdot_plus, qdot_minus, t, additional_data):
        if self.pass_additional_data:
            return self._k_potential(q_plus, q_minus, qdot_plus, qdot_minus, t, additional_data)
        else:
            return self._k_potential(q_plus, q_minus, qdot_plus, qdot_minus, t)

    @partial(jax.jit, static_argnums=(0,))
    def lagrangian_d(self, qi_values, t0, additional_data=None):
        # Eq. 4 (part 2)
        t_quadrature_values = t0 + (1 + self.xs) * self.dt / 2

        # Eq. 6. Given the values of qi we can compute the values of qidot at the quadrature points.
        qidot_vec = jax.numpy.matmul(self.dij, qi_values)

        # Eq. 7, first evaluate the function at the quadrature points, then compute the weighted sum.
        fn_i = jax.vmap(
            self.lagrangian,
            in_axes=(0, 0, 0, None)
        )(qi_values, qidot_vec, t_quadrature_values, additional_data)

        return jnp.dot(self.ws, fn_i)

    @partial(jax.jit, static_argnums=(0,))
    def k_potential_d(
            self,
            qi_plus_values,
            qi_minus_values,
            t0,
            additional_data=None
    ):
        # Eq. 4 (part 2)
        t_quadrature_values = t0 + (1 + self.xs) * self.dt / 2

        # Eq. 6. Given the values of qi we can compute the values of qidot at the quadrature points.
        qi_plus_dot_vec = jax.numpy.matmul(self.dij, qi_plus_values)
        qi_minus_dot_vec = jax.numpy.matmul(self.dij, qi_minus_values)

        # Eq. 7, first evaluate the function at the quadrature points, then compute the weighted sum.
        fn_i = jax.vmap(
            self.k_potential,
            in_axes=(0, 0, 0, 0, 0, None)
        )(
            qi_plus_values,
            qi_minus_values,
            qi_plus_dot_vec,
            qi_minus_dot_vec,
            t_quadrature_values,
            additional_data
        )

        return jnp.dot(self.ws, fn_i)

    def derivatives(self, qi_values, t0, additional_data=None):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.lagrangian_d, argnums=0)(qi_values, t0, additional_data=additional_data)

    def k_derivatives(self, qi_plus_values, qi_minus_values, t0, additional_data=None):
        # Note these arg-numbers are on the *bound* methods and hence skips the self argument
        return jax.grad(self.k_potential_d, argnums=(0, 1))(qi_plus_values, qi_minus_values, t0,
                                                            additional_data=additional_data)

    def compute_qi_values(self, previous_q, previous_pi, t_value, additional_data=None):
        optimiser_result = self._optimiser.run(
            fill_out_initial(previous_q, r=self.r - 1),
            t_value,
            previous_q,
            previous_pi,
            additional_data
        )

        return jnp.insert(optimiser_result.params, 0, previous_q, axis=0)

    def compute_pi_next(self, qi_values, t_value, additional_data=None):
        # Eq 13(b)
        dld_dqi_values = self.derivatives(qi_values, t_value, additional_data=additional_data)
        dkd_dqi_plus_values, dkd_dqi_minus_values = self.k_derivatives(qi_values,
                                                                       jnp.zeros_like(qi_values, dtype=float),
                                                                       t_value, additional_data=additional_data)
        return dld_dqi_values[-1] + dkd_dqi_minus_values[-1]

    @partial(jax.jit, static_argnums=(0,))
    def residue(self, trailing_qi_values, t, q0, pi0, additional_data=None):
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
        dld_dqi_values = self.derivatives(qi_values, t, additional_data=additional_data)

        # Evaluate in the physical limit
        dkd_dqi_plus_values, dkd_dqi_minus_values = self.k_derivatives(
            qi_values,
            jnp.zeros_like(qi_values, dtype=float),
            t,
            additional_data=additional_data
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
            t_value,
            additional_data: Any = None
    ):
        (previous_q, previous_pi) = previous_state
        qi_values = self.compute_qi_values(previous_q, previous_pi, t_value, additional_data)

        # q_{n, r + 1} = q_{n + 1, 0}
        q_next = qi_values[-1]
        pi_next = self.compute_pi_next(qi_values, t_value, additional_data)
        next_state = (q_next, pi_next)

        return next_state, next_state


class Solver:
    system: DiscretisedSystem

    def __init__(self, system: DiscretisedSystem):
        self.system = system

    def verify_args(self, q0, pi0, t0, iterations, result_orientation):
        if not (isinstance(q0, jnp.ndarray) and isinstance(pi0, jnp.ndarray)):
            raise ValueError("q0 and pi0 must be jax numpy arrays.")

        if q0.shape != pi0.shape:
            raise ValueError("q0 and pi0 must have the same shape.")

        if result_orientation not in ['time', 'coordinate']:
            raise ValueError("orientation must be either 'time' or 'coordinate'.")

        if (not isinstance(iterations, int)) and iterations >= 0:
            raise ValueError("iterations must be an integer greater than or equal to 0")

    def integrate(
            self,
            q0: jnp.ndarray,
            pi0: jnp.ndarray,
            t0: float,
            iterations: int,
            result_orientation: str = 'time',
            additional_data: Any = None
    ):
        raise NotImplementedError


class SolverScan(Solver):
    def time_domain(self, t0: float, iterations: int) -> jnp.ndarray:
        return t0 + (1 + np.arange(iterations)) * self.system.dt

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def integrate(self, q0: jnp.ndarray, pi0: jnp.ndarray, t0: float, iterations: int, result_orientation: str = 'time',
                  additional_data=None):
        print("PANDAS SolverScan.integrate recompiled")
        self.verify_args(q0, pi0, t0, iterations, result_orientation)
        print("PANDAS SolverScan.integrate verified")

        # These are the values of t which we will sample the solution at. This does not include the initial value of t
        # as the initial state of the system is already known.
        # NOTE: We use np.arange over jnp.arange as iterations is a static argument and np.arange seems to be faster.
        t_samples = self.time_domain(t0, iterations)

        print("PANDAS SolverScan.integrate t_samples computed")

        _, (q, pi) = jax.lax.scan(
            f=lambda *args: self.system.compute_next(*args, additional_data=additional_data),
            xs=t_samples,
            init=(q0, pi0),
        )

        print("PANDAS SolverScan.integrate scanned")

        q_with_initial = jnp.append(jnp.array([q0]), q, axis=0)
        pi_with_initial = jnp.append(jnp.array([pi0]), pi, axis=0)

        if result_orientation == 'time':
            return q_with_initial, pi_with_initial
        else:
            return q_with_initial.T, pi_with_initial.T


class SolverBatchedScan(Solver):
    """
    This is a version of the DiscretisedSystem which uses the batched scan function to integrate the system. This is
    expected to be faster than the standard SolverScan, while losing JIT-ability for the integrate function.

    This is good when integrating the same system to different iteration counts, as the overhead of JIT-compiling the
    integrate function is amortised over the total number of batches integrated over.
    """

    def __init__(self, system: DiscretisedSystem, batch_size: int):
        super().__init__(system)
        self.batch_size = batch_size

    @partial(jax.jit, static_argnums=(0,))
    def _integrate_inner_batch(
            self,
            carry: tuple[jnp.ndarray, jnp.ndarray],
            ts: jnp.ndarray,
            additional_data=None
    ):
        (q0, pi0) = carry

        _, (q, pi) = jax.lax.scan(
            f=lambda *args: self.system.compute_next(*args, additional_data=additional_data),
            xs=ts,
            length=self.batch_size,
            init=(q0, pi0),
        )

        return (q[-1], pi[-1]), (q, pi)

    def integrate(self, q0: jnp.ndarray, pi0: jnp.ndarray, t0: float, iterations: int, result_orientation: str = 'time',
                  additional_data=None):
        print(f"entrance, {time.time_ns()}")
        self.verify_args(q0, pi0, t0, iterations, result_orientation)

        print(f"post_t_setup_verify, {time.time_ns()}")
        number_of_batches = np.ceil(iterations / self.batch_size)
        print(f"post_t_setup_compute_n_batches, {time.time_ns()}")
        t_samples_extended = t0 + (1 + np.arange(self.batch_size * number_of_batches)) * self.system.dt
        print(f"post_t_setup_compute_t_samples_extended, {time.time_ns()}")
        t_samples_batched = t_samples_extended.reshape(-1, self.batch_size)
        print(f"post_t_setup_compute_t_samples_batched, {time.time_ns()}")

        print(f"post_t_setup, {time.time_ns()}")

        qs = []
        pis = []
        q_previous = q0
        pi_previous = pi0

        for i in range(int(number_of_batches)):
            (q_previous, pi_previous), (q, pi) = self._integrate_inner_batch(
                carry=(q_previous, pi_previous),
                ts=t_samples_batched[i],
                additional_data=additional_data
            )

            qs.append(q)
            pis.append(pi)

        print(f"post_scan, {time.time_ns()}")

        q = jnp.concatenate(qs, axis=0)
        pi = jnp.concatenate(pis, axis=0)

        print(f"post_concat_concat, {time.time_ns()}")

        q = q[:iterations]
        pi = pi[:iterations]

        print(f"post_concat_truncate, {time.time_ns()}")

        # TODO: Replace all things with jnp.append as opposed to jnp.insert
        q_with_initial = jnp.append(jnp.array([q0]), q, axis=0)
        pi_with_initial = jnp.append(jnp.array([pi0]), pi, axis=0)

        print(f"post_concat_append, {time.time_ns()}")
        print(f"post_concat, {time.time_ns()}")

        if result_orientation == 'time':
            return q_with_initial, pi_with_initial
        else:
            return q_with_initial.T, pi_with_initial.T


class SolverManual(Solver):
    """
    This is a version of the DiscretisedSystem which uses manual looping to integrate the system. This is expected to be
    much slower than both the standard SolverScan and the SolverBatchedScan, but is useful for debugging and
    understanding the system.
    """

    def integrate(
            self,
            q0: jnp.ndarray,
            pi0: jnp.ndarray,
            t0: float,
            iterations: int,
            result_orientation: str = 'time',
            additional_data=None
    ):
        self.verify_args(q0, pi0, t0, iterations, result_orientation)

        # These are the values of t which we will sample the solution at. This does not include the initial value of t
        # as the initial state of the system is already known.
        t_samples = t0 + (1 + np.arange(iterations)) * self.system.dt

        qs = [q0]
        pis = [pi0]

        carry = (q0, pi0)

        for t in t_samples:
            carry, (q_next, pi_next) = self.system.compute_next(carry, t, additional_data=additional_data)
            qs.append(q_next)
            pis.append(pi_next)

        q = jnp.stack(qs)
        pi = jnp.stack(pis)

        if result_orientation == 'time':
            return q, pi
        else:
            return q.T, pi.T
