# -*- coding: utf-8 -*-
################################################################
#                                                              #
# Tyler Bendele                                                #
# Course Number ECE351 and Section 51                          #
# Lab Number 12                                                #
# Due April 26, 2022                                           #
# Final Project                                                #
#                                                              #
################################################################

# the other packages you import will go here
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd
import scipy
from scipy.fftpack import fft, fftshift
import control as con

# load input signal
df = pd.read_csv('NoisySignal.csv')

t = df ['0']. values
sensor_sig = df ['1']. values

plt.figure ( figsize = (10 , 7) )
plt.plot (t , sensor_sig )
plt.grid ()
plt.title ('Noisy Input Signal')
plt.xlabel ('Time [s]')
plt.ylabel ('Amplitude [V]')
plt.show ()

# your code starts here , good luck
# Task 1

fs = 1e6
steps = 1e-2
ta = np.arange(1e-5, 2, steps)


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
        if abs(X_mag[i]) < 0.05:
            X_phi[i] = 0
# ----- End of user defined function ----- #    

    return freq, X_mag, X_phi

freq1, X_mag1, X_phi1 = cleanfft(sensor_sig)

def make_stem ( ax ,x ,y , color ='k', style ='solid', label ='', linewidths =2.5 ,** kwargs ) :
    ax.axhline ( x [0] , x [ -1] ,0 , color ='r')
    ax.vlines (x , 0 ,y , color = color , linestyles = style , label = label , linewidths = linewidths )
    ax.set_ylim ([1.05* y . min () , 1.05* y . max () ])


# Magnitude and Phase of Full Range
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_mag1)
plt.title('Clean FFt Magnitude of Noisy signal')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_phi1)
plt.title('Clean FFt Phase of Noisy signal')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.show()

# Magnitude and Phase of 0 to 1800 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_mag1)
plt.title('Clean FFt Magnitude from 0 to 1800')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(0,1800)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_phi1)
plt.title('Clean FFt Phase from 0 to 1800')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(0,1800)
plt.show()

# Magnitude and Phase of 1800 to 2000 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_mag1)
plt.title('Clean FFt Magnitude from 1800 to 2000')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(1800,2000)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_phi1)
plt.title('Clean FFt Phase from 1800 to 2000')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(1800,2000)
plt.show()

# Magnitude and Phase of 2000 to 100000 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_mag1)
plt.title('Clean FFt Magnitude from 2000 to 100000')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(2000,100000)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq1 , X_phi1)
plt.title('Clean FFt Phase from 2000 to 100000')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(2000,100000)
plt.show()


# Part 2 Filter Test
R1 = 10e3
R2 = 10e3
C1 = 8.85e-9      
C2 = 7.96e-9          

num1 = [(1/C2*R2), 0]
den1 = [1, ((1/(C1*R1)) + (1/(C1*R2)) + (1/(C2*R2))), (1/(C1*C2*R1*R2))]

steps = 1
w = np.arange(1e3, 1e6 +steps, steps)

sys = scipy.signal.TransferFunction( num1, den1)
w1, mag1, phase1 = scipy.signal.bode(sys, w)

plt.figure(figsize=(10,7))
plt.subplot(1,1,1)
plt.semilogx(w1, mag1)
plt.title('Magnitude Transfer Function using Bode Function')                           
plt.ylabel('|H(jw)|')
plt.xlabel('w')
plt.grid(True)

num2, den2 = scipy.signal.bilinear(num1, den1, fs)
y = scipy.signal.lfilter(num2, den2, sensor_sig)

plt.figure(figsize=(10,7))
plt.subplot(1,1,1)
plt.plot(t, y)
plt.grid(True)
plt.title('Part 2 Noisy Signal Filtered Test')                           
plt.ylabel('y(t)')
plt.xlabel('t')
plt.show()

# Part 3
sys = con.TransferFunction ( num1 , den1 )
plt.figure(figsize=(10,7))
_ = con.bode( sys , w*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure(figsize=(10,7))
_ = con.bode( sys , np.arange(1e-5, 1800 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure(figsize=(10,7))
_ = con.bode( sys , np.arange(1800, 2000 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

plt.figure(figsize=(10,7))
_ = con.bode( sys , np.arange(2000, 100000 +steps, steps)*2*np.pi , dB = True , Hz = True , deg = True , Plot = True )

# Part 4
freq2, X_mag2, X_phi2 = cleanfft(y)
# Magnitude and Phase of Full Range
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_mag2)
plt.title('Clean FFt Magnitude of Noisy signal')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_phi2)
plt.title('Clean FFt Phase of Noisy signal')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.show()

# Magnitude and Phase of 0 to 1800 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_mag2)
plt.title('Clean FFt Magnitude from 0 to 1800')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(0,1800)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_phi2)
plt.title('Clean FFt Phase from 0 to 1800')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(0,1800)
plt.show()

# Magnitude and Phase of 1800 to 2000 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_mag2)
plt.title('Clean FFt Magnitude from 1800 to 2000')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(1800,2000)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_phi2)
plt.title('Clean FFt Phase from 1800 to 2000')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(1800,2000)
plt.show()

# Magnitude and Phase of 2000 to 100000 Hz
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_mag2)
plt.title('Clean FFt Magnitude from 2000 to 100000')
plt.ylabel('|x(t)|')
plt.xlabel('f[Hz]')
plt.xlim(2000,100000)
plt.show()

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem( ax , freq2 , X_phi2)
plt.title('Clean FFt Phase from 2000 to 100000')
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt.xlim(2000,100000)
plt.show()

# Code That didn't work out
"""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

plt.subplot(ax1)
make_stem (ax1, freq1, X_mag1)
plt.title('Clean FFt Magnitude of Noisy signal')
plt.ylabel('|x(t)|')
plt . subplot ( ax2 )
make_stem (ax2, freq1, X_phi1 )
plt.ylabel('/_x(t)')
plt.xlabel('f[Hz]')
plt . show ()
"""
"""
R = 1000
L = 50e-3
C = 100e-9

num1 = [1/(R*C), 0]
den1 = [1, 1/(R*C), 1/(L*C)]
steps = 1
f = np.arange(1e3, 1e6 +steps, steps)
w = 2*np.pi*f
"""
"""
plt.figure(figsize=(12,8))
plt.subplot(3,2,(1,2))
plt.plot (t , sensor_sig )
plt.title('Task 1 - Clean FFt of Noisy signal')
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

"""