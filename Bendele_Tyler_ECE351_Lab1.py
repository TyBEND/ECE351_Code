import numpy as np
import scipy.signal as sig
import time

# printing
t = 1
print(t)
print("t =",t)
print('t =',t,"seconds")
print('t is now =',t/3,"\n...and can be rounded using 'round()'",round(t/3,4))
# exponent, in this case squared
print(3**2)
print('\n')
# lists and numpy arrays
list1 = [0,1,2,3]
print('list1:',list1)
list2 = [[0],[1],[2],[3]]
print('list2:',list2)
list3 = [[0,1],[2,3]]
print('list3:',list3)
array1 = np.array([0,1,2,3])
print('array1:',array1)
array2 = np.array([[0],[1],[2],[3]])
print('array2:',array2)
array3 = np.array([[0,1],[2,3]])
print('array3:',array3)

print('\n')
# each package can be called using
# this line prints pi
print(np.pi)

# This is a comment, and the next line shouldn't execute:
# print(t+5)

print('\n')
# large array
print(np.arange(4),'\n',
      np.arange(0,2,0.5),'\n',
      np.linspace(0,1.5,4))

print('\n')
# Indexing Lists and Arrays
list1 = [1,2,3,4,5]
array1 = np.array(list1) #definiton of a numpy array using a list
print('list1 :',list1[0],list1[4])
print('array1:',array1[0],array1[4])
array2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
list2 = list(array2)
print('array2:',array2[0,2],array2[1,4])
print('list2 :',list2[0],list2[1])

# To access an entire row or column
print('\n')
print(array2[:,2], array2[0,:])
print('\n')
# Defining a matrix as an array of zeros or ones
print('1x3:',np.zeros(3))
print('2x3:',np.zeros((2,2)))
print('2x3:',np.ones((2,3)))


print('\n','\n')
# Code showing how to use mat1potlib.pyplot
# Sample code for lab1
# Needs Numpy also, but is already imported
import matplotlib.pyplot as plt

#Define Varaiables
steps = 0.1 #step size
x = np.arange(-2,2+steps,steps) #notice the final value is
                                # '2+steps' to include '2'
y1 = x + 2
y2 = x**2

# Code for plots
plt.figure(figsize=(12,8)) # start a new figure, with
                           # a custom figure size
plt.subplot(3,1,1) # subplot 1: subplot format(row,column, number)
plt.plot(x,y1) # choosing plot variables for x and y axes
plt.title('Sample Plots for Lab 1') # title for entire figure
                                    # (all three subplots)
plt.ylabel('Subplot 1') # label for subplot 1
plt.grid(True) # show grid on plot

plt.subplot(3,1,2) #subplot 2
plt.plot(x,y2)
plt.ylabel('Subplot 2') # label for subplot 2
plt.grid(which='both') # use major and minor grids
# (minor grids not available since plot is small)

plt.subplot(3,1,3) # subplot 3
plt.plot(x,y1,'--r',label='y1')
plt.plot(x,y2,'o',label='y2') # plotting both functions on one plot
plt.axis([-2.5, 2.5, -0.5, 4.5]) # define axis
plt.grid(True)
plt.legend(loc='lower right') # prints a legned on the plot
plt.xlabel('x') # x-axis label for all three subplots (entire figure)
plt.ylabel('Subplot 3') # label for subplot 3
plt.show() ### --- This MUST be included to view your plots! ---###

# Complex numbers
print('\n')
cRect = 2 + 3j
print(cRect)

cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
print(cPol) # notice Python will store this in rectangular form

cRect2 = np.real(cPol) + 1j*np.imag(cPol)
print(cRect2) # converting from polar to rectangular

# To tell the interpreter that you want a complex number, include an addition of 0j in the sqrt
#print(np.sqrt(3*5 - 5*5))
# This is what the code should look like to make the code run correctly
print(np.sqrt(3*5 - 5*5 + 0j))

