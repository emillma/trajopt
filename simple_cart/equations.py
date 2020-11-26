# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:00:43 2020

@author: user_id
"""
import sympy as sp
from sympy.physics.mechanics import (
    dynamicsymbols,
    ReferenceFrame,
    Point,
    Particle,
    KanesMethod
)
from dynamic_system import DynamicSystem


def second_order_system():
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
    # Set the position of P0
    P0.set_pos(origin, coordinates[0] * ref_frame.x)
    P0.set_vel(ref_frame, speeds[0] * ref_frame.x)   # Set the velocity of P0
    Pa0 = Particle('Pa0', P0, m[0])             # Define a particle at P0

    # List to hold the n + 1 frames
    frames = [ref_frame]
    points = [P0]                             # List to hold the n + 1 points
    # List to hold the n + 1 particles
    particles = [Pa0]

    # List to hold the n + 1 applied forces, including the input force, f
    applied_forces = [(P0, cart_thrust * ref_frame.x - m[0] * g *
                       ref_frame.y)]
    # List to hold kinematic ODE's
    kindiffs = [coordinates[0].diff(t) - speeds[0]]

    # Initialize the object
    kane = KanesMethod(ref_frame, q_ind=coordinates,
                       u_ind=speeds, kd_eqs=kindiffs)
    # Generate EoM's fr + frstar = 0
    fr, frstar = kane.kanes_equations(particles, applied_forces)

    state = coordinates + speeds
    gain = [cart_thrust]

    kindiff_dict = kane.kindiffdict()
    M = kane.mass_matrix_full.subs(kindiff_dict)
    F = kane.forcing_full.subs(kindiff_dict)

    static_parameters = [g, m[0]]

    transfer = M.inv() * F
    return DynamicSystem(state, gain, static_parameters, transfer)
