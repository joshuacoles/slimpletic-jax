# from functools import partial
#
# import jax.lax
# import jax.numpy as jnp
# import jaxopt
# import numpy as np
# from jax import jit, grad
# from matplotlib import pyplot as plt
# from jax.experimental import checkify
#
# from slimpletic import DiscretisedSystem, SolverScan, GGLBundle
#
# q0 = jnp.array([0.0])
# pi0 = jnp.array([1.0])
# t0 = 0
# iterations = 100
# dt = 0.1
# dof = 1
# t = t0 + dt * np.arange(0, iterations + 1)
#
# ggl_bundle = GGLBundle(r=0)
#
#
# @jit
# def compute_action(state, embedding):
#     state_dof = state.size
#     return jax.lax.fori_loop(
#         0, state_dof ** 2,
#         lambda i, acc: acc + (embedding[i] * state[i // state_dof] * state[i % state_dof]),
#         0.0
#     )
#
#
# def embedded_lagrangian(q, v, t, embedding):
#     return compute_action(jnp.concat([q, v], axis=0), embedding)
#
#
# def plot_comparison(embedding):
#     actual_q, actual_pi = embedded_system_solver.integrate(
#         q0=q0,
#         pi0=pi0,
#         t0=t0,
#         iterations=iterations,
#         additional_data=embedding
#     )
#
#     plt.plot(t, expected_q)
#     plt.plot(t, actual_q, linestyle='dashed')
#     plt.show()
#
#
# # # The system which will be used when computing the loss function.
# # embedded_system_solver = SolverScan(DiscretisedSystem(
# #     dt=dt,
# #     ggl_bundle=ggl_bundle,
# #     lagrangian=embedded_lagrangian,
# #     k_potential=None,
# #     pass_additional_data=True
# # ))
# #
# # true_embedding = jnp.array([-0.5, 0, 0, 0.5])
#
#
# def rms(x, y):
#     return jnp.sqrt(jnp.mean((x - y) ** 2))
#
#
# @jit
# def loss_fn(embedding: jnp.ndarray, target_q: jnp.ndarray, target_pi: jnp.ndarray):
#     q, pi = embedded_system_solver.integrate(
#         q0=q0,
#         pi0=pi0,
#         t0=t0,
#         iterations=iterations,
#         additional_data=embedding
#     )
#
#     return jnp.sqrt(rms(q, target_q) ** 2 + rms(pi, target_pi) ** 2)
#
#
# # expected_q, expected_pi = embedded_system_solver.integrate(
# #     q0=q0,
# #     pi0=pi0,
# #     t0=t0,
# #     iterations=iterations,
# #     additional_data=jnp.array([-0.5, 0, 0, 0.5]),
# # )
#
# # results = jaxopt.GradientDescent(
# #     loss_fn,
# #     maxiter=1000,
# #     verbose=True,
# # ).run(
# #     jnp.array(np.random.rand((2 * dof) ** 2)),
# #     expected_q,
# #     expected_pi
# # ).params
#
# # plot_comparison(results)
#
#
# # Dead stupid loss function, this will say if we are using jaxopt GradientDescent class correctly.
# def dead_stupid_loss_fn(embedding: jnp.ndarray, _target_q: jnp.ndarray, _target_pi: jnp.ndarray):
#     return jnp.sqrt(jnp.mean((embedding - true_embedding) ** 2))
#
#
# # dsl_results = jaxopt.GradientDescent(
# #     dead_stupid_loss_fn,
# #     maxiter=1000,
# #     verbose=True,
# # ).run(
# #     jnp.array(np.random.rand((2 * dof) ** 2)),
# #     expected_q,
# #     expected_pi
# # ).params
#
# # plot_comparison(dsl_results)
