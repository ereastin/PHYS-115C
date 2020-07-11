#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:36:54 2020

@author: evaneastin
"""

# [PHYS 115C] HW 2 Problem 5
# shows splitting of degenerate energy states as the strength of the perturbing electric field is increased
# 'particle-on-a-ring' model

import numpy as np
import matplotlib.pyplot as plt

H_0 = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 4]]) # initial Hamiltonian for quantum numbers n = 0, 1, 2
perturbation = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]) # perturbing Hamiltonian
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
plt.title('particle on a ring in an electric field');
plt.xlabel('Perturbation strength');
plt.ylabel('Energy')
plt.grid()
plt.show()