from jax import jit


class Special:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dtype = "special"


@jit
def f(special):
    return special.x + special.y


special = Special(1, 2)

try:
    print(
        f(special))
except TypeError as e:
    # TypeError: Argument '<__main__.Special object at 0x7f1403ee5e10>' of type <class '__main__.Special'> is not a valid JAX type
    print(e)

from jax import tree_util

tree_util.register_pytree_node(Special, lambda s: ((s.x, s.y), None), lambda _aux_data, xs: Special(xs[0], xs[1]))

print(f(special))  # 3
