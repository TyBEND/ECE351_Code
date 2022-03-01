# -*- coding: utf-8 -*-
# ###############################################################
#                                                               #
# Tyler Bendele                                                 #
# Course ECE351 and Section 51                                  #  
# Lab 6                                                         #
# Due February 22, 2022                                         #
# Introduce the scipy.signal.residue() function                 #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-3
t = np.arange(1e-5, 2+steps, steps) 

def u(t):          #step function
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def impulse(t):
    y = (0.5 - 0.5*np.exp(-4*t) + np.exp(-6*t))*u(t)
    return y


num = [1, 6, 12] # Creates a matrix for the numerator
den = [1, 10, 24] # Creates a matrix for the denominator

tout , yout = sig.step(( num , den) , T = t )

# Plot tout , yout
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, impulse(t))
plt.grid()
plt.ylabel('Hand Calculated')
plt.title('Impulse Response')

plt.subplot(2, 1, 2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('Scipy Impulse Solution') 
plt.xlabel('t')
plt.show()

num2 = [0, 1, 6, 12] # Creates a matrix for the numerator
den2 = [1, 10, 24, 0] # Creates a matrix for the denominator

R, P, K = sig.residue(num2, den2)

print(R)
print(P)
print(K)

num3 = [0, 0, 0, 0, 0, 0, 25250] # Creates a matrix for the numerator
den3 = [1, 18, 218, 2036, 9085, 25250, 0] # Creates a matrix for the denominator
R, P, K = sig.residue(num3, den3)

print(R)
print(P)
print(K)

resn1 = [- 0.48557692+0.72836538j, -0.48557692-0.72836538j,
        0.09288674-0.04765193j, 0.09288674+0.04765193j]

resd1 = [-3 + 4.j,  -3. -4.j, -1.+10.j, -1.-10.j]

def cosmethod(d, n):
    y = 0
    for i in range(len(d)):
        k = np.abs(n[i])
        ka = np.angle(n[i])
        a = np.real(d[i])
        o = np.imag(d[i])
        y += k*np.exp(a*t)*np.cos(o*t + ka)
    return y

t = np.arange(0, 4.5 + steps, steps)

y = (cosmethod(resd1, resn1) + 1 - 0.21461963*np.exp(-10*t))*u(t)

# Part 2 Task 3
den4 = [1, 18, 218, 2036, 9085, 25250]
tout2, yout2 = sig.step((num3, den4), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel ('hand solved')

plt.subplot (2, 1, 2)
plt.plot (tout2, yout2)
plt.grid()
plt.ylabel('scipy solved')
plt.xlabel('t')
plt.show()
