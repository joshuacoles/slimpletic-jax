def basic_power_series(q, v, _, embedding):
    """
    A 3 parameter family of Lagrangians for 1D coordinates of the form

    \\begin{equation}
    L(q, v) = q^2 \\cdot e_0 + v^2 \\cdot e_1 + qv e_2
    \\end{equation}

    where $e \\in \\R^3$ is the embedding vector.
    """
    v = (embedding[0] * (q[0] ** 2) +
         embedding[1] * (v[0] ** 2) +
         embedding[2] * (q[0] * v[0]))

    return v


def power_series_with_prefactor(q, v, _, embedding):
    """
    A 4 parameter family of Lagrangians for 1D coordinates of the form

    \\begin{equation}
    L(q, v) = e_3 \\cdot (q^2 \\cdot e_0 + v^2 \\cdot e_1 + qv e_2)
    \\end{equation}

    where $e \\in \\R^3$ is the embedding vector.
    """
    v = embedding[3] * (embedding[0] * (q[0] ** 2) +
                        embedding[1] * (v[0] ** 2) +
                        embedding[2] * (q[0] * v[0]))

    return v
