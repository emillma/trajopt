# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:29:09 2020

@author: user_id
"""
# %%
from equations import n, m, g, length, kane, q, u, f
from sympy import Dummy, lambdify
from numpy import array, hstack, linspace, pi, ones
from numpy.linalg import solve
from scipy.integrate import odeint
from animate import animate_pendulum
from matplotlib import pyplot as plt

# The maximum length of the pendulum is 1 meter
arm_length = 1. / n
# The maximum mass of the bobs is 10 grams
bob_mass = 0.01 / n
# Parameter definitions starting with gravity and the first bob
parameters = [g, m[0]]

# Numerical values for the first two
parameter_vals = [9.81, 0.01 / n]
for i in range(n):                           # Then each mass and length
    parameters += [length[i], m[i + 1]]
    parameter_vals += [arm_length, bob_mass]

# Make a list of the states
dynamic = q + u
# Add the input force
dynamic.append(f)
# Create a dummy symbol for each variable
dummy_symbols = [Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))

# Get the solved kinematical differential equations
kindiff_dict = kane.kindiffdict()

M = kane.mass_matrix_full.subs(kindiff_dict).subs(
    dummy_dict)  # Substitute into the mass matrix

F = kane.forcing_full.subs(kindiff_dict).subs(
    dummy_dict)      # Substitute into the forcing vector

# Create a callable function to evaluate the mass matrix
M_func = lambdify(dummy_symbols + parameters, M)

# Create a callable function to evaluate the forcing vector
F_func = lambdify(dummy_symbols + parameters, F)


def right_hand_side(x, t, args):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.

    """
    u = 0.0                              # The input force is always zero
    arguments = hstack((x, u, args))     # States, input, and parameters
    dx = array(solve(M_func(*arguments),  # Solving for the derivatives
                     F_func(*arguments))).T[0]

    return dx


# Initial conditions, q and u
x0 = hstack((0, pi / 2 * ones(len(q) - 1), 1e-3 * ones(len(u))))
# Time vector
t = linspace(0, 10, 1000)
y = odeint(right_hand_side, x0, t, args=(
    parameter_vals,))         # Actual integration


# lines = plot(t, y[:, :y.shape[1] // 2])
# lab = xlabel('Time [sec]')
# leg = legend(dynamic[:y.shape[1] // 2])

# %%
fig, anim = animate_pendulum(t, y, arm_length)
plt.show()

# return fig, anim

# %%
