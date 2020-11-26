import sympy as sp
import numpy as np
from dynamic_system import DynamicSystem


def optimization_parameters(system: DynamicSystem, N=21, end_time=3):

    state = system.state
    gain = system.gain
    transfer = system.transfer
    static_parameters = system.static_parameters
    static_parameter_vals = [9.81, 1]
    static_parameter_dict = dict(zip(static_parameters, static_parameter_vals))

    dynamic = state + gain

    f_dyn = transfer.subs(static_parameter_dict)

    timestep = end_time/(N-1)

    state_shape = len(state)
    gain_shape = len(gain)
    path_constraint_min = [-3, -10]
    path_constraint_max = [3, 10]
    init_state = [0, 0]
    final_state = [1, 0]

    state_grid_points = sp.Matrix(sp.MatrixSymbol('x', N, state_shape))
    gain_grid_points = sp.Matrix(sp.MatrixSymbol('u', N-1, gain_shape))

    collocation_constraints = []
    path_constraints = []
    boundary_constraints = []

    for i in range(N-1):
        x_kp0 = state_grid_points[i, :].T
        x_kp1 = state_grid_points[i+1, :].T

        u_kp0 = gain_grid_points[i]

        f_kp0 = f_dyn.subs(dict(zip(dynamic, [*x_kp0, u_kp0])))
        f_kp1 = f_dyn.subs(dict(zip(dynamic, [*x_kp1, u_kp0])))

        if False:
            collocation_constraints.extend([*(
                timestep * (f_kp0 + f_kp1) / 2. - (x_kp1 - x_kp0))])
        if True:  # Use Hermite-Simpson Collocation
            x_kphalf = (x_kp0 + x_kp1)/2. + timestep * (f_kp0 - f_kp1) / 8.
            f_kphalf = f_dyn.subs(dict(zip(dynamic, [*x_kphalf, u_kp0])))
            collocation_constraints.extend([*(
                timestep * (f_kp0 + 4*f_kphalf + f_kp1) / 6. - (x_kp1 - x_kp0))])

    for i in range(N):
        for j in range(state_shape):
            path_constraints.append(state_grid_points[i, j]
                                    - path_constraint_min[j])

            path_constraints.append(-state_grid_points[i, j]
                                    + path_constraint_max[j])

    for i in range(state_shape):
        boundary_constraints.append(state_grid_points[0, i]
                                    - init_state[i])
        boundary_constraints.append(state_grid_points[-1, i]
                                    - final_state[i])

    args = [i for i in state_grid_points] + [i for i in gain_grid_points]

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

    cost = sum([i**2 for i in gain_grid_points])
    cost_lamda = sp.lambdify([args], cost)

    initial_states = np.linspace(init_state, final_state, N)
    initial_gains = np.zeros(N-1)
    initial_args = np.concatenate((initial_states.ravel(), initial_gains))

    return cost_lamda, initial_args, constraints
