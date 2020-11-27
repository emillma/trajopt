from get_optimization_parameters import (
    system,
    cost_lamda,
    initial_args,
    cost_jacobian_lambda,
    constraints,
    N,
    end_time
)
from scipy.optimize import minimize
from scipy.interpolate import CubicHermiteSpline
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp


def cb(a):
    pass


state_shape = system.state_shape
gain_shape = system.gain_shape

result = minimize(cost_lamda, initial_args, jac=cost_jacobian_lambda,
                  method='SLSQP', constraints=constraints, callback=cb)

states = result.x[:N*state_shape].reshape(-1, state_shape)
gains = result.x[N*state_shape:].reshape(-1, gain_shape)

# fikse dette til å løse u = B.inv()*(x_dot_final - A*x_final)

time = np.linspace(0, end_time, N)

dummy_symbols = [sp.Dummy() for i in system.dynamic_variables]
dummy_dict = dict(zip(system.dynamic_variables, dummy_symbols))

f_dyn = system.state_derivative_dyn
f_dyn_lambda = sp.lambdify([dummy_symbols], f_dyn.subs(dummy_dict))

derivatives = [f_dyn_lambda(np.concatenate((states[i], gains[i]))).ravel()
               for i in range(N)]
end_dot = f_dyn_lambda(np.concatenate((states[-1], gains[-1]))).ravel()

spline_func = CubicHermiteSpline(time, states, derivatives, axis=0)

plot_time = np.linspace(0, end_time, 100)
interpolated = spline_func(plot_time)
plt.plot(plot_time, interpolated)
plt.show()
pass
