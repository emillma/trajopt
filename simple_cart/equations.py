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
from scipy.optimize import minimize
from sympy.printing.pycode import NumPyPrinter
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    KanesMethod
)
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
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

static_parameters = [g, m[0]]
parameter_vals = [9.81, 1]
parameter_dict = dict(zip(static_parameters, parameter_vals))

f_dyn = M.inv() * F
f_dyn = f_dyn.subs(parameter_dict)

dynamic = coordinates + speeds + [cart_thrust]
dummy_symbols = [sp.Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))
f_dyn_lambda = sp.lambdify(dummy_symbols, f_dyn.subs(dummy_dict))


n = 21
end_time = 3
timestep = end_time/(n-1)
# timestep = 0.1
nlp_x = sp.Matrix(sp.MatrixSymbol('x', n, 2))
nlp_u = sp.Matrix(sp.MatrixSymbol('u', n-1, 1))
collocation_constraints = []
for i in range(n-1):
    x_kp0 = nlp_x[i, :].T
    x_kp1 = nlp_x[i+1, :].T

    u_kp0 = nlp_u[i]

    f_kp0 = f_dyn.subs(dict(zip(dynamic, [*x_kp0, u_kp0])))
    f_kp1 = f_dyn.subs(dict(zip(dynamic, [*x_kp1, u_kp0])))
    x_kphalf = (x_kp0 + x_kp1)/2. + timestep * (f_kp0 - f_kp1) / 8.
    f_kphalf = f_dyn.subs(dict(zip(dynamic, [*x_kphalf, u_kp0])))

    # collocation_constraints.extend([*(
    #     timestep * (f_kp0 + f_kp1) / 2. - (x_kp1 - x_kp0))])
    collocation_constraints.extend([*(
        timestep * (f_kp0 + 4*f_kphalf + f_kp1) / 6. - (x_kp1 - x_kp0))])

path_constraints = []
d_max = 3
u_max = 1000
for i in range(n):
    pos_k = nlp_x[i, 0]
    path_constraints.append(pos_k + d_max)
    path_constraints.append(-pos_k + d_max)

for i in range(n-1):
    u_k = nlp_u[i]
    path_constraints.append(u_k + u_max)
    path_constraints.append(-u_k + u_max)

init_pos = 0
init_vel = 0
final_pos = 1
final_vel = 0
boundary_constraints = [nlp_x[0, 0] - init_pos,
                        nlp_x[0, 1] - init_vel,
                        nlp_x[-1, 0] - final_pos,
                        nlp_x[-1, 1] - final_vel]

args = [i for i in nlp_x] + [i for i in nlp_u]

constraints = []
for con in collocation_constraints:
    fun_lambda = sp.lambdify([args], con)
    jac_lambda = sp.lambdify([args], sp.Matrix([con]).jacobian(args))
    constraints.append({'type': 'eq',
                        'fun': fun_lambda,
                        'jac': jac_lambda})

for con in path_constraints:
    fun_lambda = sp.lambdify([args], con)
    jac_lambda = sp.lambdify([args], sp.Matrix([con]).jacobian(args))
    constraints.append({'type': 'ineq',
                        'fun': fun_lambda,
                        'jac': jac_lambda})

for con in boundary_constraints:
    fun_lambda = sp.lambdify([args], con)
    jac_lambda = sp.lambdify([args], sp.Matrix([con]).jacobian(args))
    constraints.append({'type': 'eq',
                        'fun': fun_lambda,
                        'jac': jac_lambda})

lam = sp.lambdify(args, collocation_constraints)
cost = sum([i**2 for i in nlp_u])
cost_lam = sp.lambdify([args], cost)
initial_states = np.zeros((n, 2))
initial_states[:, 0] = np.linspace(0, final_pos, n)
initial_gains = np.zeros(n-1)
initial_args = np.concatenate((initial_states.ravel(), initial_gains))


def cb(a):
    pass


result = minimize(cost_lam, initial_args,
                  method='SLSQP', constraints=constraints, callback=cb)

x = result.x
pos = [x[i] for i in range(0, n*2, 2)]
vel = [x[i] for i in range(1, n*2, 2)]
gain = [x[i] for i in range(n*2, n*2+n-1)]

time = np.linspace(0, end_time, n)
f2 = CubicSpline(time, pos, bc_type=((1, vel[0]), (1, vel[-1])))
f3 = CubicSpline(time, vel, bc_type=((1, gain[0]), (1, gain[-1])))

plot_time = np.linspace(0, end_time, 100)
plt.plot(plot_time, f2(plot_time))
