# -*- coding: utf-8 -*-
# ###############################################################
#                                                               #
# Tyler Bendele                                                 #
# Course ECE351 and Section 51                                  #  
# Lab 5                                                         #
# Due February 15, 2022                                         #
# Step and Impulse Response of a RLC Band Pass Filter           #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-6
t = np.arange(0, 1.2e-3+steps, steps) 

def u(t):          #step function
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def impulse(t):
    y = -10355.6 * np.exp(-5000*t) * np.sin(18584*t + 105.06) * u(t)
    return y


num = [0, 10000, 0] # Creates a matrix for the numerator
den = [1, 10000, 3.8037e8] # Creates a matrix for the denominator

tout , yout = sig.impulse(( num , den) , T = t )

# Plot tout , yout
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, impulse(t))
plt.grid()
plt.ylabel('Hand Calculated')
plt.title('Impulse Response of a RLC Band Pass Filter')

plt.subplot(2, 1, 2)
plt.plot(tout, yout)
plt.grid()
plt.ylabel('Scipy Impulse Solution') 
plt.xlabel('t')
plt.show()

tout2 , yout2 = sig.step((num, den), T=t)
plt.figure(figsize = (10, 7))
plt.subplot(1, 1, 1)
plt.plot(tout2, yout2)
plt.grid()
plt.ylabel('Scipy Step Response') 
plt.xlabel('t')
plt.show()
