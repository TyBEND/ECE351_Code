################################################################
#                                                              #
# Tyler Bendele                                                #
# Course Number ECE351 and Section 51                          #
# Lab Number 4                                                 #
# Due Febrary 15, 2022                                         #
# This code goes over finding the step response of certain     #
# functions using convolution                                  #
#                                                              #
################################################################

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:27:12 2022

@author: Tyler Bendele
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

steps = 1e-2
t = np.arange(-10, 10 + steps, steps)
# Unit step function and convolution functions
def u(t):          #step function
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def my_conv(f1,f2):    #Convolution function
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Ext = np.append(f1, np.zeros((1, Nf2 - 1)))
    f2Ext = np.append(f2, np.zeros((1, Nf1 - 1)))
    result = np.zeros(f1Ext.shape)
    for i in range(Nf2 + Nf1 - 2):
        result[i] = 0
        for j in range(Nf1):
            if ((i - j + 1) > 0):
                try: 
                    result[i] += f1Ext[j] * f2Ext[i - j + 1]
                except:
                    print(i, j)
    return result

# First Impulse Response
def h1(t):
    a = np.exp(-2 * t)*(u(t) - u(t - 3))
    return a

# Second Impulse Response
def h2(t):
    b = u(t-2) - u(t-6)
    return b

# Third Impulse Response
def h3(t):
    w = 0.25 * 2 * np.pi
    c = np.cos(w * t)*u(t)
    return c

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t, h1(t))
plt.title('Part 1: User Defined Functions') 
                                 
plt.ylabel('h1(t)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, h2(t))
plt.ylabel('h2(t)') 
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t, h3(t))
plt.grid(True)
plt.xlabel('t') 
plt.ylabel('h3(t)') 
plt.show() 
 
# Convolutions:
y = u(t)
a = h1(t)
b = h2(t)
c = h3(t)

ch1 = my_conv(y, a)*steps
ch2 = my_conv(y, b)*steps
ch3 = my_conv(y, c)*steps

t = np.arange(-10, 10 + steps, steps)
N = len(t)
tExt = np.arange(2 * t[0], 2 * t[N-1] + steps, steps)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(tExt, ch1)
plt.grid()
plt.ylabel('h1(t)*u(t)')
plt.title('Unit Step Responses')

plt.subplot(3, 1, 2)
plt.plot(tExt, ch2)
plt.grid()
plt.ylabel('h2(t)*u(t)')

plt.subplot(3, 1, 3)
plt.plot(tExt, ch3)
plt.grid()
plt.ylabel('h3(t)*u(t)') 
plt.xlabel('t')
plt.show()

# Convolution calculation check
t = np.arange(-20, 20 + steps, steps)

def h1c(t):
    x = (0.5*(1 - np.exp(-2*t))*u(t)) - (0.5*(1 - np.exp(-2*(t-3)))*u(t-3))
    return x

def h2c(t):
    y = ((t-2)*u(t-2)) - ((t-6)*u(t-6))
    return y
def h3c(t):
    w = 0.25 * 2 * np.pi
    z = (1/w)*np.sin(w*t)*u(t)
    return z

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, h1c(t))
plt.grid()
plt.ylabel('h1(t)*u(t)')
plt.title('Unit Step Response Calculations')

plt.subplot(3, 1, 2)
plt.plot(t, h2c(t))
plt.grid()
plt.ylabel('h2(t)*u(t)')

plt.subplot(3, 1, 3)
plt.plot(t, h3c(t))
plt.grid()
plt.ylabel('h3(t)*u(t)') 
plt.xlabel('t')
plt.show()