#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:17:02 2020

@author: evaneastin
"""

# [PHYS 115C] HW 4 P 2
# adapted from Dr. Patterson's MATLAB code twoqubitsTEMPLATE.m

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


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



def plotManySweeps():
    plotSweep(2)
    plotSweep(1)
    plotSweep(0.5)
    plotSweep(0.2)


    
def plotSweepsUpandDown():
    plotSweepUpandDown(49);
    plotSweepUpandDown(50);
    plotSweepUpandDown(51);
    plotSweepUpandDown(52);


def Hamiltonianz1z2(omega1, omega2, coupling):
    Sx1 = 0.5 * np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    Sx2 = 0.5 * np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    Sy1 = 0.5 * 1j * np.array([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]])
    Sy2 = 0.5 * 1j * np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    Sz1 = 0.5 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
    Sz2 = 0.5 * np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    S1S2 = (np.matmul(Sx1, Sx2) + np.matmul(Sy1, Sy2) + np.matmul(Sz1, Sz2))
    return (omega1 * Sz1) + (omega2 * Sz2) + (coupling * S1S2)

def HamiltonianOmegaDetuning(Omega, detuning, coupling):
    omega1 = Omega + detuning / 2
    omega2 = Omega - detuning / 2
    return Hamiltonianz1z2(omega1, omega2, coupling)


def expectvalH(eigv, H):
    return np.matmul(np.matmul(eigv.conj().T, H), eigv)


def plotSweepUpandDown(sweepTime):
    Omega = 10
    coupling = 1
    timeStep = 0.01
    startDetuning = -10
    endDetuning = 10
    sweepTime = sweepTime
    times = np.linspace(0, sweepTime, 1000)
    detunings = np.linspace(startDetuning, endDetuning, int(len(times) / 2))
    detunings2 = np.linspace(endDetuning, startDetuning, int(len(times) / 2))
    totaldet = np.concatenate((detunings, detunings2))
    H_0 = HamiltonianOmegaDetuning(Omega, startDetuning, coupling)
    w, v = np.linalg.eigh(H_0)
    psi = v[1]
    a = []
    for i in range(0, len(times)):
        A = -1j * i * timeStep * HamiltonianOmegaDetuning(Omega, totaldet[i], coupling)
        psi = np.matmul(linalg.expm(A), psi)
        a.append(np.real(expectvalH(psi, HamiltonianOmegaDetuning(Omega, totaldet[i], coupling))))
    plt.figure(2)
    plt.plot(times, a, label=str(sweepTime))
    


plotManySweeps()
plt.grid()
plt.xlim(0, 50)
plt.ylim(-6, 4)
plt.title(r'Expectation Value of H as a function of time')
plt.xlabel('Sweep Time')
plt.ylabel(r'$<H>(t)$')
plt.legend(title='coupling value')
# plt.savefig('coupling.png')

plotSweepsUpandDown()
plt.grid()
plt.legend(title='sweep time')
plt.xlim(0, 55)
plt.ylim(-6, 0)
plt.title('Expectation value as a function of time')
plt.xlabel('Sweep Time')
plt.ylabel(r'$<H>(t)$')
# plt.savefig('sweeptime.png')


