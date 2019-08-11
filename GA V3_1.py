""" Using GA V2_2 code as foundation and changing the desired output to a repeated pattern """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time

"""
desired_x = []
for b in range(4) :
#5s
    for a in range(100) :
        desired_x.append(0)
    for a in range(100) :
        desired_x.append(1)
    for a in range(100) :
        desired_x.append(2)
    for a in range(100) :
        desired_x.append(3)
    for a in range(100) :
        desired_x.append(2)
"""

#"""
# Get x values of the sine wave
timearray        = np.arange(0, 20, 0.01);
# Amplitude of the sine wave is sine of a variable like time
desired_x   = np.sin(timearray)
#"""

timesteps = 20/0.01 #Simulation length set here
#Creating timestep table for step functions
times=[]
f=0.00
n = 0.01 #100Hz
for p in range(int(timesteps)):
    times.append(f)
    f = f+n

plt.plot(times, desired_x, label = "Desired Output - X_d(s)")
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude')
# Set a title of the current axes.
#plt.title('System Response to Varying Step Inputs')
plt.title('System Response over 20 Seconds.')

plt.xticks(np.arange(0, 20.5, step=2))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()