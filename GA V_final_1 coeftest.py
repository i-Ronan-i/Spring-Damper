""" Genetic Algorithm Tuning of a PID controller 
for offline optimisation of a Mass-Spring-Damper System. """

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp, RK23, odeint
from scipy.interpolate import PchipInterpolator
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time
import matplotlib.pyplot as plt

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
pop = [[44.65, 443.8, 845.52]] #specially coverged Nicola one
timesteps = 20/0.01 #Simulation length set here
desired_x = []

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

#Creating timestep table for step functions
times=[]
f=0.00
n = 0.01 #100Hz
for p in range(int(timesteps)):
    times.append(f)
    f = f+n

"""
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
"""



def setpoint(t):
    if (t > 0 and t < 4):
            r = 2
    elif (t > 4 and t < 9):
            r = 3
    elif (t > 9 and t < 14):
            r = 8
    elif (t > 14 and t < 21):
            r = 3
    else:
            r = 0

    return r

tempo = np.linspace(1e-05, 20, 20*100)
rise_time = 0.1
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))

set_interp = PchipInterpolator(tempo, set_point)

rise_time = 0.1
a0 = 1  # M  [Kg]
a1 = 10  # C  [Ns/m]
a2 = 20  # K  [N/m]
r_max = max(desired_x)
r_min = min(desired_x)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max

force_constraint = a0 * ar + a1 * vr + a2 * yr
print("Force constraint is (N): ", force_constraint)

def sys2PID(t,x):
    global force_constraint
    Kp = 443.8  # Proportional Gain
    Ki =  845.52 # Integrative Gain
    Kd = 44.65  # Derivative Gain

    r=set_interp(t)

    #State Variables
    y = x[0]            # x1 POSITION
    dydt = x[1]         # x2 VELOCITY
    yi = x[2]           # x3

    u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output

    if abs(u)>force_constraint:
        print("A L E R T")

    dxdt = [0,0,0]

    dxdt[0] = dydt
    dxdt[1] = (- a1 * dydt - a2 * y + u)/a0
    dxdt[2] = r - y

    return [dxdt[0],dxdt[1],dxdt[2]]

tev = np.linspace(0,20,1000)

x_ini = [0,0,0]        # initial conditions
solga = solve_ivp(sys2PID, [0, 20], x_ini, t_eval=tev)
yga = solga.y[0, :]
tga = solga.t






#Plotting code
plt.plot(tempo,set_point,"r--",label="set point command [m]")
#plt.plot(times, y_out, label = "System Output - X(s)") 
plt.plot(tga, yga, label="System Response - X(s)")
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude')
# Set a title of the current axes.
plt.title('System Response to Sin(t) over 20 Seconds')

plt.xticks(np.arange(0, 20.5, step=1))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()