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
#"""
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
#"""
"""
def setpoint(t):
    if (t >= 0 and t < 1) or (t>=5 and t<6) or (t>=10 and t<11) or (t>=15 and t<16):
            r = 0
    elif (t >= 1 and t < 2) or (t>=6 and t<7) or (t>=11 and t<12) or (t>=16 and t<17):
            r = 1
    elif (t >= 2 and t < 3) or (t>=7 and t<8) or (t>=12 and t<13) or (t>=17 and t<18):
            r = 2
    elif (t >= 3 and t < 4) or (t>=8 and t<9) or (t>=13 and t<14) or (t>=18 and t<19):
            r = 2.5
    elif (t >= 4 and t < 5) or (t>=9 and t<10) or (t>=14 and t<15) or (t>=19 and t<20):
            r = 1
    else:
            r = 0
    return r
#"""
tempo = np.linspace(0, 20, 20*5)
rise_time = 0.1
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))

set_interp = PchipInterpolator(tempo, set_point)

pop = [14.76, 452.59, 416.12] #GA vs GP chromosome. Time = 114.72minutes
#pop = [] #GA no constraint
#pop = [] #GA with constraint

Kp = 452.59
Ki = 416.11
Kd = 14.76
dt = 0.02
rise_time = 0.1

m = 1  # M  [Kg]
c = 5  # C  [Ns/m]
k = 10  # K  [N/m]

force_constraint = 3000 #N

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

temp = round(0.00, 2)
tev = []
for times in range(int(20/dt)):
    tev.append(temp)
    temp = round(temp + dt, 2)
x_ini = [0,0,0]        # initial conditions
solga = solve_ivp(sys2PID, [0, 20], x_ini, t_eval=tev)
y_out = solga.y[0, :]
t_out = solga.t

err_val = 0.0
for y in range(len(t_out)) :
    err_val = err_val + abs(set_interp(t_out[y]) - y_out[y])
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