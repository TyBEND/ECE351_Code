# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Tyler Bendele                                                #
# Course Number ECE351 and Section 51                          #
# Lab Number 9                                                 #
# Due March 22, 2022                                           #
# Fast Fourier Transform                                       #
#                                                              #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, fftshift

steps = 1e-3
t = np.arange(1e-5, 2, steps)
fs = 100

def fastft(x):
    N = len( x ) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
                                                 # to the center of the spectrum
    freq = np.arange (-N/2 , N/2) * fs / N # compute the frequencies for the output
                                           # signal , (fs is the sampling frequency and
                                           # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    X_phi = np.angle( X_fft_shifted ) # compute the phases of the signal
# ----- End of user defined function ----- #    
    return freq, X_mag, X_phi

# Task 1
x1 = np.cos(2*np.pi*t)
freq1, X_mag1, X_phi1 = fastft(x1)


plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x1)
plt.title('Task 1 - User Definited FFT of x1(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freq1, X_mag1)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freq1, X_mag1) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freq1, X_phi1)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freq1, X_phi1)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task 2
x2 = 5*np.sin(2*np.pi*t)
freq2, X_mag2, X_phi2 = fastft(x2)


plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x2)
plt.title('Task 2 - User Definited FFT of x2(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freq2, X_mag2)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freq2, X_mag2) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freq2, X_phi2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freq2, X_phi2)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task 3
x3 = 2*np.cos((2*np.pi*6*t) - 2) + (np.sin((2*np.pi*6*t) + 3) ** 2) 
freq3, X_mag3, X_phi3 = fastft(x3)


plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x3)
plt.title('Task 2 - User Definited FFT of x3(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freq3, X_mag3)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freq3, X_mag3) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freq3, X_phi3)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freq3, X_phi3)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task4
def cleanfft(x):
    N = len( x ) # find the length of the signal
    X_fft = scipy.fftpack.fft(x) # perform the fast Fourier transform (fft)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) # shift zero frequency components
                                                 # to the center of the spectrum
    freq = np.arange (-N/2 , N/2) * fs / N # compute the frequencies for the output
                                           # signal , (fs is the sampling frequency and
                                           # needs to be defined previously in your code
    X_mag = np.abs( X_fft_shifted ) / N # compute the magnitudes of the signal
    X_phi = np.angle( X_fft_shifted ) # compute the phases of the signal
    
    for i in range(len(X_phi)):
        if abs(X_mag[i]) < 1e-10:
            X_phi[i] = 0
# ----- End of user defined function ----- #    
    return freq, X_mag, X_phi

# Task 1 Clean
freqc1, X_magc1, X_phic1 = cleanfft(x1)

plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x1)
plt.title('Task 1 - Clean User Definited FFT of x1(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freqc1, X_magc1)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freqc1, X_magc1) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freqc1, X_phic1)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freqc1, X_phic1)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task 2 Clean
freqc2, X_magc2, X_phic2 = cleanfft(x2)


plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x2)
plt.title('Task 2 - Clean User Definited FFT of x2(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freqc2, X_magc2)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freqc2, X_magc2) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freqc2, X_phic2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freqc2, X_phic2)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task 3 Clean
freqc3, X_magc3, X_phic3 = cleanfft(x3)

plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x3)
plt.title('Task 2 - Clean User Definited FFT of x3(t)')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freqc3, X_magc3)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freqc3, X_magc3) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freqc3, X_phic3)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freqc3, X_phic3)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()

# Task 5
t = np.arange(0, 16, steps)
def ak(k):
    a = np.zeros(t.shape)
    return a

def bk(k):
    b = np.zeros(t.shape)
    b = 2/(k*np.pi) * (1 - np.cos(k*np.pi))
    return b

def x(t,k):
    T = 8
    for i in range(k):
        if i == 0:
            x = (1/2)*ak(i)
        else:
            x += ak(i) + bk(i)*np.sin((i*2*np.pi*t)/T)
    return x 

x15 = x(t,15)
freqc15, X_magc15, X_phic15 = cleanfft(x15)

plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot(t,x15)
plt.title('Task 2 - Clean User Definited FFT of Square Wave from Lab 8')
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.grid(True)

plt.subplot(3,2,3)
plt.stem(freqc15, X_magc15)
plt.ylabel('|x(t)|') 
plt.grid(which='both')

plt.subplot(3,2,4)
plt.stem(freqc15, X_magc15) 
plt.xlim(-2,2)
plt.grid(which='both')

plt.subplot(3,2,5)
plt.stem(freqc15, X_phic15)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 

plt.subplot(3,2,6)
plt.stem(freqc15, X_phic15)
plt.xlim(-2,2)
plt.xlabel('f[Hz]') 
plt.ylabel('/_x(t)') 
plt.show()
