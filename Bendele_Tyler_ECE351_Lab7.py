# -*- coding: utf-8 -*-
# ###############################################################
#                                                               #
# Tyler Bendele                                                 #
# Course ECE351 and Section 51                                  #  
# Lab 7                                                         #
# Due March 1, 2022                                             #
# Block Diagrams and system stability                           #
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


num = [0, 0, 1, 9] # Creates a matrix for the numerator
den = [1, -2, -40, -64] # Creates a matrix for the denominator

z, p, k= sig.tf2zpk(num, den)
print(z)
print(p)
print(k)

num2 = [0, 1, 4] # Creates a matrix for the numerator
den2 = [1, 4, 3] # Creates a matrix for the denominator

z, p, k= sig.tf2zpk(num2, den2)
print(z)
print(p)
print(k)

num3 = [1, 26, 168] # Creates a matrix for the numerator

b = np.roots(num3)
print(b)

num4 = sig.convolve([1,9] , [1,4])
print('Numerator =', num4)

den4 = sig.convolve([1, -2, -40, -64] , [1, 4, 3])
print('Denominator =', den4)

tout, yout = sig.step((num4, den4), T=t)

plt.figure(figsize = (10, 7))
plt.plot(tout, yout)
plt.grid()
plt.ylabel('Step response')
plt.xlabel('t')
plt.title('Open Loop Step Response')
plt.show()

numA = [1, 4]
numG = [1, 9]
numB = [1, 26, 168]
denA = [1, 4, 3]
denG = [1, -2, -40, -64]

num5 = sig.convolve(numA, numG)
print(num5)

den5 = sig.convolve(denA, denG)
print(den5)

num6 = sig.convolve(numB, numG)
den6 = sig.convolve(num6, denA)
print(den6)

den7 = den5 + den6
print(den7)

z, p, k= sig.tf2zpk(num5, den7)
print(z)
print(p)
print(k)

tout2, yout2 = sig.step((num5, den7), T=t)

plt.figure(figsize = (10, 7))
plt.plot(tout2, yout2)
plt.grid()
plt.ylabel('Step response')
plt.xlabel('t')
plt.title('Closed Loop Step Response')
plt.show()





