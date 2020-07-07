#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:58:50 2020

@author: evaneastin
"""

import numpy as np

m = np.array([[-1/4, 1/2], [1/2, -1/4]])
w, v = np.linalg.eigh(m)
print(w, v)

import matplotlib.pyplot as plt

delta = np.linspace(-10, 10, 1000)
omega = 10
epsilon = 1
hbar = 1.054 * 10**(-34)

E_1 = hbar * (epsilon/4 + omega)
E_2= (hbar/2) * (delta - 3 * epsilon / 2)
E_3 = (hbar/2) * (epsilon / 2 - delta)
E_4 = hbar * (epsilon / 4 - omega)

plt.hlines(E_1, -10, 10, color='r', label='E_1')
plt.plot(delta, E_2, label='E_2')
plt.plot(delta, E_3, label='E_3')
plt.hlines(E_4, -10, 10, label='E_4')
plt.legend()
plt.grid()
plt.show()