# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:17:49 2020

@author: user_id
"""
import numpy as np
import pprint


pp = pprint.PrettyPrinter(indent=4)


def rk4(y_n, t_n, h, f):
    '''
    Gir y_n+1 ved hjelp av Runge Kutta 4.
    '''
    # Konstanter
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + (h/2)*k1)
    k3 = f(t_n + h/2, y_n + (h/2)*k2)
    k4 = f(t_n + h,   y_n + h * k3)

    # Regn ut og returner neste verdi
    return y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

class StateModel:
    def __init__(self):
        self.state = np.array([-1,0], dtype = np.float64)
        self.time = 0

    def f(self, time, state):
        return np.array([[0, -10], [1, 0]]) @ state

    def __str__(self):
        pp = pprint.PrettyPrinter(indent=4)
        return pp.pformat(self.state)

    def __repr__(self):
        return self.__str__()

    def iterate(self, delta = 0.01):
        self.state = rk4(self.state, self.time, delta, self.f)
        self.time += delta
        return self.state[0], self.time


if __name__ == '__main__':
    sm = StateModel()
    sm.iterate()
