import sympy as sp
import numpy as np
from simpson_integral import get_integral_square


class DynamicSystem():
    def __init__(self, state_variables, gain_variables, static_parameters,
                 state_derivative):

        self.state_variables = state_variables

        self.gain_variables = gain_variables

        self.dynamic_variables = self.state_variables + self.gain_variables

        self.state_derivative = sp.Matrix(state_derivative)

        self.static_parameters = static_parameters

        self.state_shape = len(self.state_variables)
        self.gain_shape = len(self.gain_variables)

    def set_static_values(self, static_values):
        assert len(static_values) == len(self.static_parameters)
        self.static_parameter_values = static_values

        self.static_variable_dict = dict(
            zip(self.static_parameters, static_values)
        )
        self.state_derivative_dyn = self.state_derivative.subs(
            self.static_variable_dict)

    def set_grid_size(self, N, end_time):
        self.N = N
        self.end_time = end_time
        self.timestep = self.end_time / (self.N - 1)

        self.state_grid_points = sp.Matrix(
            sp.MatrixSymbol('X', N, self.state_shape))

        self.gain_grid_points = sp.Matrix(
            sp.MatrixSymbol('u', N*2-1, self.gain_shape))

        self.optimization_args = (list(self.state_grid_points)
                                  + list(self.gain_grid_points))

    def set_boundary_constraints(self, init_state, final_state):
        self.init_state = init_state
        self.final_state = final_state
        self.boundary_constraints = []
        for i in range(self.state_shape):
            if init_state[i] is not None:
                self.boundary_constraints.append(
                    self.state_grid_points[0, i] - init_state[i])
            if final_state[i] is not None:
                self.boundary_constraints.append(
                    self.state_grid_points[-1, i] - final_state[i])

        initial_states = np.zeros(self.state_grid_points.shape)
        for i in range(self.state_shape):
            if init_state[i] is not None and final_state[i] is not None:
                initial_states[:, i] = np.linspace(init_state[i],
                                                   final_state[i], self.N)

            elif init_state[i] is not None:
                initial_states[:, i] = init_state[i]

            elif final_state[i] is not None:
                initial_states[:, i] = final_state[i]

        initial_gains = np.zeros(self.gain_grid_points.shape)

        self.initial_optimization_args = np.concatenate(
            (initial_states.ravel(), initial_gains.ravel()))

    def set_collocation_constraints(self, use_hermite_himpson=True):
        self.collocation_constraints = []
        for i in range(self.N-1):
            x_kp0 = self.state_grid_points[i, :].T
            x_kp1 = self.state_grid_points[i+1, :].T

            u_kp0 = self.gain_grid_points[i*2, :]
            u_kp1 = self.gain_grid_points[(i+1)*2, :]
            u_kphalf = self.gain_grid_points[i*2 + 1, :]

            f_kp0 = self.state_derivative_dyn.subs(
                dict(zip(self.dynamic_variables, [*x_kp0, *u_kp0])))
            f_kp1 = self.state_derivative_dyn.subs(
                dict(zip(self.dynamic_variables, [*x_kp1, *u_kp1])))

            if use_hermite_himpson:
                x_kphalf = ((x_kp0 + x_kp1)/2.
                            + (self.timestep / 8.) * (f_kp0 - f_kp1))
                f_kphalf = self.state_derivative_dyn.subs(
                    dict(zip(self.dynamic_variables, [*x_kphalf, *u_kphalf])))
                self.collocation_constraints.extend([*(
                    (self.timestep / 6.) * (f_kp0 + 4*f_kphalf + f_kp1)
                    - (x_kp1 - x_kp0))])

            else:
                self.collocation_constraints.extend([*(
                    (self.timestep / 2.) * (f_kp0 + f_kp1)
                    - (x_kp1 - x_kp0))])

    def set_path_constraints(self, state_min, state_max, gain_min, gain_max):
        self.state_min = state_min
        self.state_max = state_max
        self.gain_min = gain_min
        self.gain_max = gain_max

        self.path_constraints = []
        for i in range(self.N):
            for j in range(self.state_shape):
                if state_min[j] is not None:
                    self.path_constraints.append(self.state_grid_points[i, j]
                                                 - state_min[j])
                if state_max[j] is not None:
                    self.path_constraints.append(-self.state_grid_points[i, j]
                                                 + state_max[j])
        for i in range(2*self.N - 1):
            for j in range(self.gain_shape):
                if gain_min[j] is not None:
                    self.path_constraints.append(self.gain_grid_points[i, j]
                                                 - gain_min[j])
                if gain_max[j] is not None:
                    self.path_constraints.append(-self.gain_grid_points[i, j]
                                                 + gain_max[j])

    def set_cost_function(self):
        cost = 0
        for i in range(self.N-1):
            u_kp0 = self.gain_grid_points[i*2, :]
            u_kp1 = self.gain_grid_points[(i+1)*2, :]
            u_kphalf = self.gain_grid_points[i*2 + 1, :]
            cost += get_integral_square(
                1, u_kp0[0], u_kphalf[0], u_kp1[0])
            # cost += get_integral_square(
            #     1, u_kp0[1], u_kphalf[1], u_kp1[1])

        cost_jacobian = sp.Matrix([cost]).jacobian(self.optimization_args)
        self.cost_function = sp.lambdify([self.optimization_args], cost)
        self.cost_jacobian_function = sp.lambdify(
            [self.optimization_args], cost_jacobian)

    def get_constraint_functions(self):
        constraints_functions = []
        for con in self.collocation_constraints:
            fun_lambda = sp.lambdify([self.optimization_args], con)
            jac_lambda = sp.lambdify([self.optimization_args], sp.Matrix(
                [con]).jacobian(self.optimization_args))
            constraints_functions.append({'type': 'eq',
                                          'fun': fun_lambda,
                                          'jac': jac_lambda})

        for con in self.path_constraints:
            fun_lambda = sp.lambdify([self.optimization_args], con)
            jac_lambda = sp.lambdify([self.optimization_args], sp.Matrix(
                [con]).jacobian(self.optimization_args))
            constraints_functions.append({'type': 'ineq',
                                          'fun': fun_lambda,
                                          'jac': jac_lambda})

        for con in self.boundary_constraints:
            fun_lambda = sp.lambdify([self.optimization_args], con)
            jac_lambda = sp.lambdify([self.optimization_args], sp.Matrix(
                [con]).jacobian(self.optimization_args))
            constraints_functions.append({'type': 'eq',
                                          'fun': fun_lambda,
                                          'jac': jac_lambda})

        return tuple(constraints_functions)

    def augment_state_grid_point(self, state_grid_points,
                                 gain_grid_points):
        augmented_state_grid = []
        for i in range(self.N-1):
            x_kp0 = state_grid_points[i, :].T
            x_kp1 = state_grid_points[i+1, :].T

            u_kp0 = gain_grid_points[i*2, :]
            u_kp1 = gain_grid_points[(i+1)*2, :]
            u_kphalf = gain_grid_points[i*2 + 1, :]

            f_kp0 = self.state_derivative_dyn.subs(
                dict(zip(self.dynamic_variables, [*x_kp0, *u_kp0])))
            f_kp1 = self.state_derivative_dyn.subs(
                dict(zip(self.dynamic_variables, [*x_kp1, *u_kp1])))

            x_kphalf = (sp.Matrix(x_kp0 + x_kp1)/2.
                        + (self.timestep / 8.) * (f_kp0 - f_kp1))

            augmented_state_grid.append(sp.Matrix(x_kp0))
            augmented_state_grid.append(x_kphalf)
        augmented_state_grid.append(x_kp1)
        return np.array([list(i) for i in augmented_state_grid])

    def get_optimization_parameters(self):
        return (self.cost_function, self.initial_optimization_args,
                self.cost_jacobian_function,
                self.get_constraint_functions())
