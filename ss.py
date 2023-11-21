import jax


class Expr:
    pass


class Add(Expr):
    lhs: Expr
    rhs: Expr

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class Mul(Expr):
    lhs: Expr
    rhs: Expr

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class Var(Expr):
    index: int

    def __init__(self, index):
        self.index = index


class Const(Expr):
    value: float

    def __init__(self, value):
        self.value = value


def compose(expr):
    if isinstance(expr, Add):
        return lambda a, b: compose(expr.lhs)(a, b) + compose(expr.rhs)(a, b)
    elif isinstance(expr, Mul):
        return lambda a, b: compose(expr.lhs)(a, b) * compose(expr.rhs)(a, b)
    elif isinstance(expr, Var):
        if expr.index == 0:
            return lambda a, b: a
        elif expr.index == 1:
            return lambda a, b: b
        else:
            raise ValueError(f"Unknown index {expr.index}")
    elif isinstance(expr, Const):
        return lambda a, b: expr.value
    else:
        raise ValueError(f"Unknown expression {expr}")


a = compose(Add(Mul(Var(1), Var(0)), Var(1)))
b = jax.jit(a)

jax.debug.print("{}", a(2., 2.))
jax.debug.print("{}", b(-1., 2.))
jax.debug.print("{}", jax.grad(a)(1., 2.))
jax.debug.print("{}", jax.vmap(jax.grad(b))(jax.numpy.array([1., 2.3, .42]), jax.numpy.array([2., 53.2, 4.6])))
