""" Genetic Algorithm Tuning of a PID controller 
for offline optimisation of a Mass-Spring-Damper System. """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time


""" MAIN CODE PROGRAM BODY """
#first thing is to create the system itself - MSD, input
#MSD state definitions
m = 1 # kg   - mass
c = 10 # Ns/m - damping coefficient
k = 20 # N/m  - spring coefficient

# Transfer function definitions
#step_mag = 1    #Magnitude of the step function - manupulation of functions, not actual step function
MSDnum = [0, 0, 1]
MSDden = [m, c, k]
#pop = [[98.47, 500.0, 900.0]] #original top performing
pop = [[46.71, 475.0, 900.0]] #specially coverged Nicola one
timesteps = 20/0.01 #Simulation length set here
desired_x = []


#Create the desired output plot for lsim
#"""
#20s for creating the amplitude values for the desired values of the system
for b in range(1) :
#5s
    for a in range(400) :
        desired_x.append(2)
    for a in range(500) :
        desired_x.append(4)
    for a in range(500) :
        desired_x.append(8)
    for a in range(600) :
        desired_x.append(3)
    #for a in range(100) :
    #    desired_x.append(4)
    """
#"""
"""
# Get x values of the sine wave
timearray        = np.arange(0, 20, 0.01)
# Amplitude of the sine wave is sine of a variable like time
desired_x   = np.sin(timearray)
#"""

#Creating timestep table for step functions
times=[]
f=0.00
n = 0.01 #100Hz
for p in range(int(timesteps)):
    times.append(f)
    f = f+n

#Create transfer functions for use by inputting current pop
GAnum = [pop[0][0], pop[0][1], pop[0][2]]   # kd, kp, ki 
GAden = [0, 1, 0]
GHs_num = signal.convolve(GAnum, MSDnum)
GHs_den = signal.convolve(GAden, MSDden)

#Create CLTF
cltf_num = GHs_num
cltf_den = GHs_den + GHs_num
cltf = signal.TransferFunction(cltf_num, cltf_den)

#Simulates the current system [s].
t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)
err_vall = 0.0
for y in range(len(times)) :
    err_vall = err_vall + abs(desired_x[y] - y_out[y])
err_val = err_vall  
print("Fitness value of top performing member: ", round(err_val, 4))

#Plotting code
plt.plot(times, desired_x, label = "Desired Output - X_d(s)")
plt.plot(times, y_out, label = "System Output - X(s)") 
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude')
# Set a title of the current axes.
plt.title('System Response over 20 Seconds: Re-Tuned PID.')

plt.xticks(np.arange(0, 20.5, step=2))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()