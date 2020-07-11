#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:11:14 2020

@author: evaneastin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def plotManySweeps():
    plotSweep(2)
    plotSweep(1)
    plotSweep(0.5)
    plotSweep(0.2)
    
def plotSweepsUpandDown():
    plotSweepUpandDown(49)
    plotSweepUpandDown(50)
    plotSweepUpandDown(51)
    plotSweepUpandDown(52)
    
def plotSweep(coupling):
    Omega = 10
    coupling = coupling
    timeStep = 0.01
    startDetuning = -10
    endDetuning = 10
    sweepTime = 50
    times = np.linspace(0, sweepTime, 100)
    detunings = np.linspace(startDetuning, endDetuning, len(times))
    H_0 = HamiltonianOmegaDetuning(Omega, startDetuning, coupling)
    w, v = np.linalg.eigh(H_0)
    psi = v[1]
    a = []
    for i in range(0, len(times)):
        A = -1j * i * timeStep * HamiltonianOmegaDetuning(Omega, detunings[i], coupling)
        psi = np.matmul(linalg.expm(A), psi)
        a.append(np.real(expectvalH(psi, HamiltonianOmegaDetuning(Omega, detunings[i], coupling))))
    plt.figure(1)
    plt.plot(times, a, label=coupling)

def HamiltonianOmegaDetuning(Omega, detuning, coupling):
    omega1 = Omega + detuning / 2
    omega2 = Omega - detuning / 2
    return Hamiltonianz1z2(omega1, omega2, coupling)

def expectvalH(eigv, H):
    return np.matmul(np.matmul(eigv.conj().T, H), eigv)
    
def Hamiltonianz1z2(omega1, omega2, coupling):
    Sx1 = 0.5 * np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    Sx2 = 0.5 * np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    Sy1 = 0.5 * 1j * np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]])
    Sy2 = 0.5 * 1j * np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    Sz1 = 0.5 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    Sz2 = 0.5 * np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    S1S2 = (np.matmul(Sx1, Sx2) + np.matmul(Sy1, Sy2) + np.matmul(Sz1, Sz2))
    return (omega1 * Sz1) + (omega2 * Sz2) + (coupling * S1S2)
  
def plotSweepUpandDown(sweepTime):
    Omega = 10
    coupling = 1
    timeStep = 0.01
    startDetuning = -10
    endDetuning = 10
    sweepTime = sweepTime
    times = np.arange(0, sweepTime, timeStep)
    times = times[0:2*round(len(times)/2)]
    numSteps = len(times)
    detunings = np.linspace(startDetuning, endDetuning, int(numSteps / 2))
    detunings = np.concatenate((detunings, np.flipud(detunings)))    
    H_0 = HamiltonianOmegaDetuning(Omega, startDetuning, coupling)
    w, v = np.linalg.eigh(H_0)
    psi = v[1]
    energies = times * 0
    for i in range(0, numSteps):
        H = HamiltonianOmegaDetuning(Omega, detunings[i], coupling)
        A = -1j * i * timeStep * H
        psi = np.matmul(linalg.expm(A), psi)
        energies[i] = np.real(expectvalH(psi, H))
    plt.figure(2)
    plt.plot(times, energies, label=sweepTime)
    
def HamiltonianBzBx(Bz, Bx):
    H = (Bz * np.array([[-1, 0], [0, 1]]) + Bx * np.array([[0, 1], [1, 0]])) / 2
    return H


def plotRabiflops(detuning):
    Bx0 = .7
    Bz = 10
    t = np.linspace(0, 51, 10000)
    timeStep = 0.01
    omegaf = 2
    Bx = Bx0 * np.cos((omegaf + detuning) * t)
    H_0 = HamiltonianBzBx(Bz, Bx0)
    w, v = np.linalg.eigh(H_0)
    psi = v[0]
    energies = t * 0
    for i in range(len(Bx)):
        H = HamiltonianBzBx(Bz, Bx[i])
        A = -1j * timeStep * H
        psi = np.matmul(linalg.expm(A), psi)
        energies[i] = np.real(expectvalH(psi, H))
    plt.figure(3)
    plt.plot(t, energies, label='Energy')
    plt.figure(4)
    plt.plot(t, Bx, label='Bx-field')
        
    
plotManySweeps()
plotSweepsUpandDown()


plotRabiflops(10)
plt.grid()
plt.legend()
plt.show()