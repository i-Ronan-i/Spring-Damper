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
pop = [[44.65, 443.8, 845.52]] #specially coverged Nicola one
Kp = 443.80
Ki = 845.52
Kd = 44.65
timesteps = 20/0.01 #Simulation length set here


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
tempo = np.linspace(0, 20, 20*50)
rise_time = 0.1
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))

set_interp = PchipInterpolator(tempo, set_point)

rise_time = 0.1
m = 1  # M  [Kg]
c = 10  # C  [Ns/m]
k = 20  # K  [N/m]
r_max = max(set_point)
r_min = min(set_point)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max

force_constraint = m * ar + c * vr + k * yr
print("Force constraint is (N): ", force_constraint)

def sys2PID(t,x):
    global force_constraint
    #Kp = 443.8  # Proportional Gain
    #Ki =  845.52 # Integrative Gain
    #Kd = 44.65  # Derivative Gain

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
    dxdt[1] = (- c * dydt - k * y + u)/m
    dxdt[2] = r - y

    return [dxdt[0],dxdt[1],dxdt[2]]

tev = np.linspace(0,20,1000)

x_ini = [0,0,0]        # initial conditions
solga = solve_ivp(sys2PID, [0, 20], x_ini, t_eval=tev)
y_out = solga.y[0, :]
t_out = solga.t

err_vall = 0.0
for y in range(len(t_out)) :
    err_vall = err_vall + abs(set_point[y] - y_out[y])
err_val = err_vall  
print("Fitness value of top performing member: ", round(err_val, 4))

#Plotting code
plt.plot(tempo,set_point,"r--",label="set point command [m]")
#plt.plot(times, y_out, label = "System Output - X(s)") 
plt.plot(t_out, y_out, label="System Response - X(s)")
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