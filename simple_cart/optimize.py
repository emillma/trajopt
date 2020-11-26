from optimization_parameters import optimization_parameters
from equations import second_order_system
from scipy.optimize import minimize
from scipy.interpolate import CubicHermiteSpline
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp


system = second_order_system()

N = 21
end_time = 3
(cost_lamda, initial_args, constraints
 ) = optimization_parameters(system, N, end_time)


def cb(a):
    pass


state_shape = system.state_shape
gain_shape = system.gain_shape

result = minimize(cost_lamda, initial_args,
                  method='SLSQP', constraints=constraints, callback=cb)

states = result.x[:N*state_shape].reshape(-1, state_shape)
gains = result.x[N*state_shape:].reshape(-1, gain_shape)

# fikse dette til å løse u = B.inv()*(x_dot_final - A*x_final)
gains = np.vstack((gains, np.zeros_like(gains[-1])))

time = np.linspace(0, end_time, N)

dynamic = system.state + system.gain
dummy_symbols = [sp.Dummy() for i in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))

static_parameters = system.static_parameters
static_parameter_vals = [9.81, 1]
static_parameter_dict = dict(zip(static_parameters, static_parameter_vals))

f_dyn = system.transfer.subs(system.static_parameter_dict)
f_dyn_lambda = sp.lambdify([dummy_symbols], f_dyn.subs(dummy_dict))

derivatives = [f_dyn_lambda(np.concatenate((states[i], gains[i]))).ravel()
               for i in range(N)]
end_dot = f_dyn_lambda(np.concatenate((states[-1], gains[-1]))).ravel()

spline_func = CubicHermiteSpline(time, states, derivatives, axis=0)

plot_time = np.linspace(0, end_time, 100)
interpolated = spline_func(plot_time)
plt.plot(plot_time, interpolated)
