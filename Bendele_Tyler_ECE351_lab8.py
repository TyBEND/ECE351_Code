# -*- coding: utf-8 -*-
# ###############################################################
#                                                               #
# Tyler Bendele                                                 #
# Course ECE351 and Section 51                                  #  
# Lab 8                                                         #
# Due March 8, 2022                                             #
# Fourier Series Approximation of a Square Wave                 #
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-3
t = np.arange(1e-5, 20+steps, steps) 

def ak(k):
    a = np.zeros(t.shape)
    return a

def bk(k):
    b = np.zeros(t.shape)
    b = 2/(k*np.pi) * (1 - np.cos(k*np.pi))
    return b

a0 = ak(0)
print("a0 =", a0)

a1 = ak(1)
print("a1 =", a1)

b1 = bk(1)
print("b1 =", b1)

b2 = bk(2)
print("b2 =", b2)

b3 = bk(3)
print("b3 =", b3)

def x(t,k):
    T = 8
    for i in range(k):
        if i == 0:
            x = (1/2)*ak(i)
        else:
            x += ak(i) + bk(i)*np.sin((i*2*np.pi*t)/T)
    return x 
    
x1 = x(t,1)
x3 = x(t,3)
x15 = x(t,15)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.title('Fourier Series Approximation (N = 1, 3, 15)')
plt.ylabel('N = 1')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,x3)
plt.ylabel('N = 3') 
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,x15)
plt.grid(True)
plt.xlabel('t') 
plt.ylabel('N = 15') 
plt.show() 

x50 = x(t,50)
x150 = x(t,150)
x1500 = x(t,1500)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t,x50)
plt.title('Fourier Series Approximation (N = 50, 150, 1500)')
plt.ylabel('N = 50')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,x150)
plt.ylabel('N = 150') 
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,x1500)
plt.grid(True)
plt.xlabel('t') 
plt.ylabel('N = 1500') 
plt.show()
