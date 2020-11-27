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
from scipy.interpolate import CubicHermiteSpline, PPoly
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp


def cb(a):
    pass


state_shape = system.state_shape
gain_shape = system.gain_shape

result = minimize(cost_lamda, initial_args, jac=cost_jacobian_lambda,
                  method='SLSQP', constraints=constraints, callback=cb,
                  options={'maxiter': 1000})

states = result.x[:N*state_shape].reshape(-1, state_shape)
gains = result.x[N*state_shape:].reshape(-1, gain_shape)
states_augmented = system.augment_state_grid_point(states, gains)
# fikse dette til å løse u = B.inv()*(x_dot_final - A*x_final)

time = np.linspace(0, end_time, N*2-1)

dummy_symbols = [sp.Dummy() for i in system.dynamic_variables]
dummy_dict = dict(zip(system.dynamic_variables, dummy_symbols))

f_dyn = system.state_derivative_dyn
f_dyn_lambda = sp.lambdify([dummy_symbols], f_dyn.subs(dummy_dict))

derivatives = np.array([f_dyn_lambda(
    np.concatenate((states_augmented[i], gains[i]))).ravel()
    for i in range(N*2-1)])

plot_time = np.linspace(0, end_time, N*20 + 1)
state_spline = CubicHermiteSpline(time, states_augmented, derivatives, axis=0)
states_interpolated = state_spline(plot_time)

quadratic_polys = np.array([np.polyfit(time[i:i+3] - time[i], gains[i:i+3], 2)
                            for i in range(0, 2*N-2, 2)]).swapaxes(0, 1)

gain_spline = PPoly(quadratic_polys, time[::2])
gain_interpolated = gain_spline(plot_time)
# plt.plot(time, states_augmented)
fig, ax = plt.subplots(1, 2)
ax[0].plot(plot_time, state_spline(plot_time))
# ax[0].plot(plot_time, state_spline.derivative()(plot_time))
# ax[0].plot(plot_time, state_spline.antiderivative()(plot_time))
ax[0].plot(time, states_augmented, 'o-')
# ax[0].plot(time, derivatives, 'o-')

ax[1].plot(plot_time, gain_interpolated)
# ax[1].plot(time, gains)
plt.show()
pass
