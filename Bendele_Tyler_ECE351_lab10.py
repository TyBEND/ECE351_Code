# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Tyler Bendele                                                #
# Course Number ECE351 and Section 51                          #
# Lab Number 10                                                #
# Due March 29, 2022                                           #
# Frequency Response                                           #
#                                                              #
################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy
import control as con # this package is not included in the Anaconda
                      # distribution, but shoould have been installed in lab 0

# Part 1 Task 1
steps = 1
w = np.arange(1e3, 1e6 +steps, steps)

def hm(w, R, L, C):
    m = (w/(R*C)) / np.sqrt((w**4) + ((((1/(R*C))**2) - (2/L*C))*(w**2)) + ((1/(L*C))**2))
    m = 20*np.log10(m)
    return m


def hp(w, R, L, C):
    a = np.pi/2 - np.arctan(((w/(R*C)))/((1/(L*C)) - (w**2)))
    a = np.degrees(a)
    for i in range(len(w)):
        if a[i] > 90:
            a[i] = a[i]-180
    return a

R = 1000
L = 27e-3
C = 100e-9

h1m = hm(w, R, L, C)
h1p = hp(w, R, L, C)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.semilogx(w, h1m)
plt.title('Magnitude and Phase of Transfer Function')                           
plt.ylabel('|H(jw)|')
plt.grid(True)

plt.subplot(2,1,2)
plt.semilogx(w, h1p)
plt.grid(True)
plt.xlabel('w') 
plt.ylabel('/_H(jw)') 
plt.show()

# P1 Task 2
sys = scipy.signal.TransferFunction( [1/(R*C), 0], [1, 1/(R*C), 1/(L*C)])
w1, mag1, phase1 = scipy.signal.bode(sys, w)

plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.semilogx(w1, mag1)
plt.title('Magnitude and Phase of Transfer Function using Bode Function')                           
plt.ylabel('|H(jw)|')
plt.grid(True)

plt.subplot(2,1,2)
plt.semilogx(w1, phase1)
plt.grid(True)
plt.xlabel('w') 
plt.ylabel('/_H(jw)') 
plt.show()

# P1 Task 3
num1 = [1/(R*C), 0]
den1 = [1, 1/(R*C), 1/(L*C)]

sys = con.TransferFunction ( num1 , den1 )
_ = con.bode( sys , w , dB = True , Hz = True , deg = True , Plot = True )
# use _ = ... to suppress the output

# Part 2 task 1
fs = 50000
steps = 1/50000
t = np.arange(0, 0.01 + steps, steps)

x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

plt.figure(figsize=(10,7))
plt.subplot(1,1,1)
plt.plot(t, x)
plt.grid(True)
plt.title('Three Frequency Plot')                           
plt.ylabel('x(t)')
plt.xlabel('t')
plt.show()

# P2 Tasks 2, 3, and 4

num2, den2 = scipy.signal.bilinear(num1, den1, fs)
y = scipy.signal.lfilter(num2, den2, x)

plt.figure(figsize=(10,7))
plt.subplot(1,1,1)
plt.plot(t, y)
plt.grid(True)
plt.title('Three Frequency Plot with lfilter')                           
plt.ylabel('x(t)')
plt.xlabel('t')
plt.show()