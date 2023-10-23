import jax.core
import jaxopt

print(jaxopt.ScipyRootFinding(
    optimality_fun=lambda x: x ** 2 - 2,
    method='hybr'
).run(1.0))

def dij(i, j):
    jax.lax.switch(
        jax.lax.eq(i, j),
        lambda: 0,
        lambda: 1
    )

jax.numpy.fromfunction(lambda i, j: i + j, (2,2))