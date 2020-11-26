import sympy as sp
import numpy as np


class DynamicSystem():
    def __init__(self, state_variables, gain_variables, static_parameters,
                 state_derivative):

        self.state_variables = state_variables
        self.gain_variables = gain_variables

        self.dynamic_variables = state_variables + gain_variables

        self.state_derivative_all = state_derivative
        self.state_derivative = state_derivative

        self.static_parameters = static_parameters

        self.state_shape = len(state_variables)
        self.gain_shape = len(gain_variables)

    def set_static_values(self, static_values):
        assert len(static_values) == len(self.static_parameters)
        self.static_parameter_values = static_values

        self.static_variable_dict = dict(
            zip(self.static_parameters, static_values)
        )
        self.state_derivative = self.state_derivative_all.subs(
            self.static_variable_dict)
        pass

    def set_grid_size(self, N, end_time):
        self.N = N
        self.end_time = end_time
        self.timestep = self.end_time / self.N

        self.state_grid_points = sp.Matrix(
            sp.MatrixSymbol('X', N, self.state_shape))

        self.gain_grid_points = sp.Matrix(
            sp.MatrixSymbol('u', N, self.gain_shape))

        self.optimization_args = (list(self.state_grid_points)
                                  + list(self.gain_grid_points))

    def set_boundary_constraints(self, init_state, final_state):
        self.init_state = init_state
        self.final_state = final_state
        self.boundary_constraints = []
        for i in range(self.state_shape):
            self.boundary_constraints.append(self.state_grid_points[0, i]
                                             - init_state[i])
            self.boundary_constraints.append(self.state_grid_points[-1, i]
                                             - final_state[i])

        initial_states = np.linspace(init_state, final_state, self.N)
        initial_gains = np.zeros(self.N)
        self.initial_optimization_args = np.concatenate(
            (initial_states.ravel(), initial_gains))

    def set_collocation_constraints(self, use_hermite_himpson=True):
        self.collocation_constraints = []
        for i in range(self.N-1):
            x_kp0 = self.state_grid_points[i, :].T
            x_kp1 = self.state_grid_points[i+1, :].T

            u_kp0 = self.gain_grid_points[i, :]
            u_kp1 = self.gain_grid_points[i+1, :]

            f_kp0 = self.state_derivative.subs(
                dict(zip(self.dynamic_variables, [*x_kp0, *u_kp0])))
            f_kp1 = self.state_derivative.subs(
                dict(zip(self.dynamic_variables, [*x_kp1, *u_kp1])))

            if use_hermite_himpson:
                x_kphalf = ((x_kp0 + x_kp1)/2.
                            + self.timestep * (f_kp0 - f_kp1) / 8.)
                u_kphalf = (u_kp0 + u_kp1) / 2.
                f_kphalf = self.state_derivative.subs(
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
        for i in range(self.N - 1):
            for j in range(self.gain_shape):
                if gain_min[j] is not None:
                    self.path_constraints.append(self.gain_grid_points[i, j]
                                                 - gain_min[j])
                if gain_max[j] is not None:
                    self.path_constraints.append(-self.gain_grid_points[i, j]
                                                 + gain_max[j])

    def set_cost_function(self):
        cost = sum([i**2 for i in self.gain_grid_points])
        self.cost_function = sp.lambdify([self.optimization_args], cost)

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

    def get_optimization_parameters(self):
        return (self.cost_function, self.initial_optimization_args,
                self.get_constraint_functions())
