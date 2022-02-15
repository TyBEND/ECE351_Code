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
y = u(t)

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
    a = np.zeros(t.shape)
    a = np.exp(2*t)*u(1 - t)
    return a
a = h1(t)

# Second Impulse Response
def h2(t):
    b = np.zeros(t.shape)
    b = u(t-2) - u(t-6)
    return b
b = h2(t)

# Third Impulse Response

def h3(t):
    c = np.zeros(t.shape)
    for i in range(len(t)):
        c[i] = np.cos(2*np.pi*0.25*t[i])
    return c
c = h3(t)*u(t)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t,a)
plt.title('Part 1: User Defined Functions') 
                                 
plt.ylabel('h1(t)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,b)
plt.ylabel('h2(t)') 
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,c)
plt.grid(True)
plt.xlabel('t') 
plt.ylabel('h3(t)') 
plt.show() 
 

t = np.arange(-10, 10 + steps, steps)
N = len(t)
tExt = np.arange(0, 2 * t[N-1], steps)

x = my_conv(a, y) * steps
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tExt, x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('Convolution of f1 and f2')
plt.show()

