from matplotlib.pyplot import plot
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicHermiteSpline, PPoly

N = 3
time = np.linspace(0, 3, 2*N+1)
gains = np.random.random((2*N+1, 2))

quadratic_polys = np.array([np.polyfit(time[:3], gains[i:i+3], 2)
                            for i in range(0, 2*N-1, 2)]).swapaxes(0, 1)

# quadratic_polys = np.flip(quadratic_polys, 0)
i = 0
poly1 = np.polyfit(time[i:i+3], gains[i:i+3], 2)
i = 2
poly2 = np.polyfit(time[i:i+3], gains[i:i+3], 2)
i = 4
poly3 = np.polyfit(time[i:i+3], gains[i:i+3], 2)

gain_spline = PPoly(quadratic_polys, time[::2])
plot_time = np.linspace(0, 3, 100)


# plt.plot(plot_time, np.polyval(poly1[:, 0], plot_time))
# plt.plot(plot_time, np.polyval(poly2[:, 0], plot_time))
# plt.plot(plot_time, np.polyval(poly3[:, 0], plot_time))
plt.plot(time, gains[:, 0])
plt.plot(plot_time, gain_spline(plot_time))
plt.show()
