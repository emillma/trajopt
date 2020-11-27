import sympy as sp

"""
integral of (u(t))**2 from 0 to h
38:40 in https://www.youtube.com/watch?v=wlkRYMVUZTs&t=1379s
"""
h = sp.symbols('h')
u0, uhalf, u1 = sp.symbols('u0 uhalf u1')
beta1 = (-1/h) * (3 * u0 - 4*uhalf + u1)
beta2 = (2/h**2) * (u0 - 2*uhalf + u1)

delta = sp.symbols('delta')
u = u0 + delta * beta1 + delta**2 * beta2
cost = sp.simplify(sp.integrate(u**2, (delta, 0, h)))


def get_integral_square(*args):
    """
    args = [h, u0, uhalf, u1]
    """
    return cost.subs(zip([h, u0, uhalf, u1], args))
