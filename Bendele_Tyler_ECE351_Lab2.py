# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams.update({'fontsize': 14}) #sets font size

steps = 1e-2
t = np.arange(0, 10 + steps, steps)

def func1(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y
y = func1(t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 1 graph: y = cos(t)')

t = np.arange(-5, 10 + steps, steps)

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

y = u(t)   
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 2: Step Function')

y = r(t)   
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 2: Ramp Function')

def part2(t):
    z = np.zeros(t.shape)
    z = r(t) - r(t-3) + 5*u(t-3) - 2*u(t-6) - 2*r(t-6)
    return z

z = part2(t)   
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, z)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 2 graph')

# Time Reversal
t = np.arange(-10, 5 + steps, steps)
tr = part2(-t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, tr)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Reversal')

# Time Shift f(t-4)
t = np.arange(0, 14 + steps, steps)
ts = part2(t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, ts)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Shift f(t-4)')

# Time Shift f(-t-4)
t = np.arange(-14, 0 + steps, steps)
ts = part2(-t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, ts)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Shift f(-t-4)')

# Time Scale f(t/2)
t = np.arange(-5, 20 + steps, steps)
tsc = part2(t/2)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, tsc)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Scale f(t/2)')

# Time Scale f(2t)
t = np.arange(-5, 10 + steps, steps)
tsc = part2(2*t)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, tsc)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Scale f(t/2)')

# Derivative
t = np.arange(-5, 10 + steps, steps)
dt = np.diff(t)
dy = np.diff(part2(t))/dt

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.ylim((-2,10))
plt.plot(t[range(len(dy))], dy)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Derivative')

"""
y = r(t) - r(t - 3) + 5 * u(t - 3) - 2 * u(t - 6) - 2 * r(t - 6)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Part 2 graph')

y = r(-t - 2) - r(-t - 5) + 5 * u(-t - 5) - 2 * u(-t - 8) - 2 * r(-t - 8)
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('time reversal')
"""