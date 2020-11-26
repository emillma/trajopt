# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:00:43 2020

@author: user_id
"""
from numpy import matrix
from sympy import symbols
import sympy as sp
import numpy as np
from scipy.integrate import odeint
import control
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    KanesMethod
)
# from sympy.printing.pycode import NumPyPrinter, pycode
q = dynamicsymbols('q:1')  # Generalized coordinates
u = dynamicsymbols('u:1')  # Generalized speeds
f = dynamicsymbols('f')                # Force applied to the cart

m = sp.symbols('m:1')         # Mass of each bob
g, t = sp.symbols('g t')                  # Gravity and time

ref_frame = ReferenceFrame('I')     # Inertial reference frame
origin = Point('O')                 # Origin point
origin.set_vel(ref_frame, 0)        # Origin's velocity is zero


P0 = Point('P0')                            # Hinge point of top link
P0.set_pos(origin, q[0] * ref_frame.x)      # Set the position of P0
P0.set_vel(ref_frame, u[0] * ref_frame.x)   # Set the velocity of P0
Pa0 = Particle('Pa0', P0, m[0])             # Define a particle at P0


# List to hold the n + 1 frames
frames = [ref_frame]
points = [P0]                             # List to hold the n + 1 points
particles = [Pa0]                         # List to hold the n + 1 particles

# List to hold the n + 1 applied forces, including the input force, f
forces = [(P0, f * ref_frame.x - m[0] * g *
           ref_frame.y)]
kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's


# Initialize the object
kane = KanesMethod(ref_frame, q_ind=q, u_ind=u, kd_eqs=kindiffs)
# Generate EoM's fr + frstar = 0
fr, frstar = kane.kanes_equations(particles, forces)

dynamic = q + u + [f]
# Add the input force
# Create a dummy symbol for each variable
dummy_symbols = [sp.Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))

# Get the solved kinematical differential equations
kindiff_dict = kane.kindiffdict()

M = kane.mass_matrix_full.subs(kindiff_dict).subs(
    dummy_dict)  # Substitute into the mass matrix

F = kane.forcing_full.subs(kindiff_dict).subs(
    dummy_dict)

parameters = [g, m[0]]
parameter_vals = [9.81, 1]
M_func = sp.lambdify(dummy_symbols + parameters, M)
F_func = sp.lambdify(dummy_symbols + parameters, F)

M, MA, MB, U = kane.linearize()
# sub in the equilibrium point and the parameters
parameter_dict = dict(zip(parameters, parameter_vals))
M = M.subs(parameter_dict)
MA = MA.subs(parameter_dict)
MB = MB.subs(parameter_dict)

M_inv = M.inv()
A = M_inv * MA
B = M_inv * MB

K = control.acker(A, B, [-1, -1])


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
    u = np.array(-K@(x-np.array([1, 0]))
                 ).squeeze()                            # The input force is always zero
    arguments = np.hstack((x, u, args))     # States, input, and parameters
    dx = np.array(np.linalg.solve(M_func(*arguments),  # Solving for the derivatives
                                  F_func(*arguments))).T[0]

    return dx


kane.linearize()
x0 = np.hstack((0, 0))
# Time vector
t = np.linspace(0, 10, 1000)
y = odeint(right_hand_side, x0, t, args=(
    parameter_vals,))         # Actual integration
