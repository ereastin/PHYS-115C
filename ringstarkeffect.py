#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:36:54 2020

@author: evaneastin
"""

# [PHYS 115C] HW 2 Problem 5

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x**2)

plt.plot(x, y, label=r'$\sin(x^{2}$')
plt.legend()
plt.grid()
# plt.savefig('115C_hw2_p5_b.png')
plt.show()

m = np.array([[0, 1], [2, 3]])

w, v = np.linalg.eig(m)
print(w)

H_0 = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 4]])
perturbation = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
Lz = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])

pValues = np.linspace(0, 5, 50);
allEigenvalues = np.zeros((len(pValues), 5))

for i in np.arange(0, len(pValues)):
    thisH = H_0  + (perturbation * pValues[i])
    w, v = np.linalg.eigh(thisH)
    allEigenvalues[i] = w
  
eigfinal = allEigenvalues
        
plt.plot(pValues, eigfinal.T[0])
plt.plot(pValues, eigfinal.T[1])
plt.plot(pValues, eigfinal.T[2])
plt.plot(pValues, eigfinal.T[3])
plt.plot(pValues, eigfinal.T[4])
plt.title('rings in an electric field');
plt.xlabel('Perturbation strength');
plt.ylabel('Energy')
plt.grid()
#plt.savefig('115C_hw2_p5_d.png')
plt.show()