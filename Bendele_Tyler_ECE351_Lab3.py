################################################################
#                                                              #
# Tyler Bendele                                                #
# Course Number ECE351 and Section 51                          #
# Lab Number 3                                                 #
# Due Febrary 8, 2022                                          #
# This code goes over how to use convolution                   #
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
t = np.arange(0, 20 + steps, steps)

def u(t):          #step function
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def r(t):          #ramp function
    y = np.zeros(t.shape)
    for i in range(len(t)):
       if t[i] > 0:
           y[i] = t[i]
       else:
           y[i] = 0
    return y

def f1(t):
    a = np.zeros(t.shape)
    a = u(t-2) - u(t-9)
    return a
a = f1(t)

def f2(t):
    b = np.zeros(t.shape)
    b = np.exp(-t)*u(t)
    return b
b = f2(t)

def f3(t):
    c = np.zeros(t.shape)
    c = r(t-2) * (u(t-2) - u(t-3)) + r(4-t) * (u(t-3) - u(t-4))
    return c
c = f3(t)

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(t,a)
plt.title('Part 1: User Defined Functions') 
                                 
plt.ylabel('f1(t)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t,b)
plt.ylabel('f2(t)') 
plt.grid(which='both')

plt.subplot(3,1,3)
plt.plot(t,c)
plt.grid(True)
plt.xlabel('t') 
plt.ylabel('f3(t)') 
plt.show() 

def my_conv(f1,f2):
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


t = np.arange(0, 20 + steps, steps)
N = len(t)
tExt = np.arange(0, 2 * t[N-1], steps)


x = my_conv(a, b) * steps

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tExt, x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('Convolution of f1 and f2')
plt.show()

y = my_conv(a, c) * steps

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tExt, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Convolution of f1 and f3')
plt.show()

z = my_conv(b, c) * steps

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tExt, z)
plt.grid()
plt.ylabel('z(t)')
plt.xlabel('t')
plt.title('Convolution of f2 and f3')
plt.show()

