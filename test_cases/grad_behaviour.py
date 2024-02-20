from jax import grad, jit

counter = 0

def f(x, y):
    global counter
    counter += 1

    print("Called {} times".format(counter))
    print("x: ", x)
    print("y: ", y)
    print()
    return x + y

print("JIT Example")
jf = jit(f)
jf(1.0, 2.0)
jf(2.0, 2.0)
jf(2.0, 2.0)
grad(jf, argnums=(0,))(1.0, 2.0)

print("\n")

print("GRAD Example")
ff = grad(f, argnums=(0,))
ff(1.0, 2.0)
ff(1.0, 2.0)
ff(3.0, 2.0)

"""
From this example we can see crucially that grad(f) evaluates the function f for each call, tracing the wrt argument,
but passing in concrete values for the other arguments.

JIT on the other hand evaluates only once with all traced arguments. 
"""

print("NOW G")

counter_g = 0

def g(x, y):
    global counter_g
    counter_g += 1

    print("Called G {} times".format(counter_g))
    print("x: ", x)
    print("y: ", y)
    print()
    return jf(x, y) * 2 + 4

jg = jit(g)
print("1st G")
jg(1.0, 2.0)
print("2nd G")
jg(1.0, 3.0)