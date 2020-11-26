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
from sympy.printing.pycode import NumPyPrinter
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    KanesMethod
)
# from sympy.printing.pycode import NumPyPrinter, pycode
coordinates = dynamicsymbols('q:1')  # Generalized coordinates
speeds = dynamicsymbols('u:1')  # Generalized speeds
# Force applied to the cart
cart_thrust = dynamicsymbols('thrust')

m = sp.symbols('m:1')         # Mass of each bob
g, t = sp.symbols('g t')                  # Gravity and time

ref_frame = ReferenceFrame('I')     # Inertial reference frame
origin = Point('O')                 # Origin point
origin.set_vel(ref_frame, 0)        # Origin's velocity is zero


P0 = Point('P0')                            # Hinge point of top link
P0.set_pos(origin, coordinates[0] * ref_frame.x)      # Set the position of P0
P0.set_vel(ref_frame, speeds[0] * ref_frame.x)   # Set the velocity of P0
Pa0 = Particle('Pa0', P0, m[0])             # Define a particle at P0


# List to hold the n + 1 frames
frames = [ref_frame]
points = [P0]                             # List to hold the n + 1 points
particles = [Pa0]                         # List to hold the n + 1 particles

# List to hold the n + 1 applied forces, including the input force, f
forces = [(P0, cart_thrust * ref_frame.x - m[0] * g *
           ref_frame.y)]
# List to hold kinematic ODE's
kindiffs = [coordinates[0].diff(t) - speeds[0]]


# Initialize the object
kane = KanesMethod(ref_frame, q_ind=coordinates, u_ind=speeds, kd_eqs=kindiffs)
# Generate EoM's fr + frstar = 0
fr, frstar = kane.kanes_equations(particles, forces)

dynamic = coordinates + speeds + [cart_thrust]

dummy_symbols = [sp.Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))

# Get the solved kinematical differential equations
kindiff_dict = kane.kindiffdict()

M = kane.mass_matrix_full.subs(kindiff_dict)

F = kane.forcing_full.subs(kindiff_dict)

parameters = [g, m[0]]
parameter_vals = [9.81, 1]
parameter_dict = dict(zip(parameters, parameter_vals))

f_dyn = M.inv() * F
f_dyn = f_dyn.subs(parameter_dict)

dynamic = coordinates + speeds + [cart_thrust]
dummy_symbols = [sp.Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))
f_dyn_lambda = sp.lambdify(dummy_symbols, f_dyn.subs(dummy_dict))

n = 11
timestep = sp.symbols('h')
timestep = 0.1
nlp_x = [sp.symbols(f'x_:{n}\,:2')[i:i+2] for i in range(0, n*2, 2)]
nlp_u = sp.symbols(f'u_:{n}')
constraints = []
for i in range(n-1):
    x_kp0 = sp.Matrix(nlp_x[i])
    x_kp1 = sp.Matrix(nlp_x[i+1])

    u_kp0 = nlp_u[i]
    u_kp1 = nlp_u[i+1]

    f_kp0 = f_dyn.subs(dict(zip(dynamic, [*x_kp0, u_kp0])))
    f_kp1 = f_dyn.subs(dict(zip(dynamic, [*x_kp1, u_kp1])))
    constraints.extend([*(
        timestep * (f_kp0 + f_kp1) / 2. - (x_kp1 - x_kp0))])

args = [nlp_x, nlp_u]
lam = sp.lambdify(args, constraints)
