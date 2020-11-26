# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:00:43 2020

@author: user_id
"""
from sympy import symbols
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    KanesMethod
)
# from sympy.printing.pycode import NumPyPrinter, pycode
n = 1

q = dynamicsymbols('q:' + str(n + 1))  # Generalized coordinates
u = dynamicsymbols('u:' + str(n + 1))  # Generalized speeds
f = dynamicsymbols('f')                # Force applied to the cart

m = symbols('m:' + str(n + 1))         # Mass of each bob
length = symbols('l:' + str(n))        # Length of each link
g, t = symbols('g t')                  # Gravity and time

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
           ref_frame.y - 0.01 * q[0] * ref_frame.x)]
kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's

for i in range(n):
    # Create a new frame
    Bi = ref_frame.orientnew('B' + str(i), 'Axis', [q[i + 1], ref_frame.z])
    # Set angular velocity
    Bi.set_ang_vel(ref_frame, u[i + 1] * ref_frame.z)
    # Add it to the frames list
    frames.append(Bi)

    Pi = points[-1].locatenew('P' + str(i + 1), length[i]
                              * Bi.x)  # Create a new point
    # Set the velocity
    Pi.v2pt_theory(points[-1], ref_frame, Bi)
    # Add it to the points list
    points.append(Pi)

    # Create a new particle
    Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])
    # Add it to the particles list
    particles.append(Pai)

    # Set the force applied at the point
    forces.append((Pi, -m[i + 1] * g * ref_frame.y - 0.001 * u[i + 1] * Bi.y))

    # Set the force applied at the point
    forces.append((points[-2],   0.001 * u[i + 1] * Bi.y))

    # Define the kinematic ODE:  dq_i / dt - u_i = 0
    kindiffs.append(q[i + 1].diff(t) - u[i + 1])


# Initialize the object
kane = KanesMethod(ref_frame, q_ind=q, u_ind=u, kd_eqs=kindiffs)
# Generate EoM's fr + frstar = 0
fr, frstar = kane.kanes_equations(particles, forces)
