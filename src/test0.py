# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:54:50 2020

@author: emilm

https://www.youtube.com/watch?v=lWbeuDwYVto&t=438s
"""

from sympy import init_printing
from sympy.abc import c, d, e, f, g, h
from sympy.physics.vector import ReferenceFrame

N = ReferenceFrame('N')

init_printing()
a = c * N.x, + d * N.y + e * N.z