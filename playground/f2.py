from jax import grad, jit, numpy as jnp


def lagrangian(x, y, x_dot, y_dot, t):
    return 0.5 * (x_dot ** 2 + y_dot ** 2) - 9.81 * y


def lagrangian_d(
        x_0,
        x_1,
        x_2,
        x_3,
        y_0,
        y_1,
        y_2,
        y_3,
        t
):
    return lagrangian(x_0, y_0, x_1, y_1, t)


def index(dof, i, ndof=2, r=2):
    return dof * ndof + i


def eom_1(ld, dof=2, r=2):
    def eom(qi, pi, t):
        x_dot = 1  # todo
        y_dot = 1  # todo

        jnp.array([
            # 13(a) x
            pi[0] + grad(ld, index(0, 0))(qi[0], qi[1], x_dot, y_dot, t),
            # 13(a) y
            pi[1] + grad(ld, index(1, 0))(q[0], q[1], x_dot, y_dot, t),

            # 13(c) x
            grad(ld, index(0, 1))(q[0], q[1], x_dot, y_dot, t),
            grad(ld, index(0, 2))(q[0], q[1], x_dot, y_dot, t),

            # 13(c) y
            grad(ld, index(1, 1))(q[0], q[1], x_dot, y_dot, t),
            grad(ld, index(1, 2))(q[0], q[1], x_dot, y_dot, t),
        ])

    return jit(eom)
