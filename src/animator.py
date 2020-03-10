# -*- coding: utf-8 -*-
"""


===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.
"""

# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from state_model import StateModel
import time



if 'fig' not in globals():

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)
    ax.axis('equal')
    ax.axis([-3, 3, -3, 3])
    ax.grid()

    def handle_close(evt):
        print('Closed Figure!')
        global fig
        del(fig)

    fig.canvas.mpl_connect('close_event', handle_close)

else:
    ax.clear()
    ax.axis([-3, 3, -3, 3])
    ax.grid()

cart, = ax.plot([], [], '-', lw=5, c='b')
pole, = ax.plot([], [], 'o-', lw=5, c='r')
time_template = 'time = %.3fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    # cart.set_data([0,1], [0,0])
    # pole.set_data([0,0], [0,-1])
    # time_text.set_text('')
    # ax.clear()
    return cart, pole, time_text


def animate(i, model):
    x, time = model.iterate()
    cart.set_data([x-0.5, x + 0.5], [0, 0])
    pole.set_data([x, x], [0, -1])
    time_text.set_text(time_template % (time))
    return cart, pole, time_text


try:
    eval('ani.event_source.stop()')
except Exception as e:
    pass

model = StateModel()
ani = FuncAnimation(fig, animate, frames=np.arange(1000),
                    interval=10, blit = True, init_func=None,
                    fargs=(model,))

plt.show()
