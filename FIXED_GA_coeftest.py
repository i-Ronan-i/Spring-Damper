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
"""
###################################
def setpoint(t):
    if (t >= 0 and t < 1) or (t>=5 and t<6) or (t>=10 and t<11) or (t>=15 and t<16):
            r = 1
    elif (t >= 1 and t < 2) or (t>=6 and t<7) or (t>=11 and t<12) or (t>=16 and t<17):
            r = 1
    elif (t >= 2 and t < 3) or (t>=7 and t<8) or (t>=12 and t<13) or (t>=17 and t<18):
            r = 2
    elif (t >= 3 and t < 4) or (t>=8 and t<9) or (t>=13 and t<14) or (t>=18 and t<19):
            r = 10
    elif (t >= 4 and t < 5) or (t>=9 and t<10) or (t>=14 and t<15) or (t>=19 and t<21):
            r = 1
    else:
            r = 0
    return r

tempo = np.linspace(0, 20, 20*10)
rise_time = 0.1
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))
#"""
#"""
tempo = np.linspace(0, 20, 20*10)
set_point = 150*np.sin(tempo)
#"""
set_interp = PchipInterpolator(tempo, set_point)


dt = 0.02
temp = round(0.00, 2)
tev = []
for times in range(int(20/dt)):
    tev.append(temp)
    temp = round(temp + dt, 2)

m = 1  # M  [Kg]
c = 5  # C  [Ns/m]
k = 10  # K  [N/m]
x_ini = [0,0,0]        # initial conditions
#pop = [14.76, 452.59, 416.12] #GA vs GP chromosome. Time = 114.72minutes
pop_nc = [26.17, 600.0, 182.29] #GA no constraint  #Time 32.49mins #Fitness 42.3859  #SinFitness 33.36
pop_c = [13.7, 598.50, 252.41] #GA with constraint #Time 50.95mins #Fitness 50.4420  #SinFitness 20.6521

Kp_nc = pop_nc[1]
Ki_nc = pop_nc[2]
Kd_nc = pop_nc[0]
Kp_c = pop_c[1]
Ki_c = pop_c[2]
Kd_c = pop_c[0]

force_constraint = 3000 #N
print("Force constraint is (N): ", force_constraint)

def sys2PID_nc(t,x):
    global force_constraint
    r=set_interp(t)

    #State Variables
    y = x[0]            # x1 POSITION
    dydt = x[1]         # x2 VELOCITY
    yi = x[2]           # x3

    u = Kp_nc * (r - y) + Ki_nc * yi - Kd_nc * dydt         #PID output

    if abs(u)>force_constraint:
        print("GA No Contraint - invalid alert")

    dxdt = [0,0,0]

    dxdt[0] = dydt
    dxdt[1] = (- c * dydt - k * y + u)/m
    dxdt[2] = r - y

    return [dxdt[0],dxdt[1],dxdt[2]]

def sys2PID_c(t,x):
    global force_constraint
    r=set_interp(t)

    #State Variables
    y = x[0]            # x1 POSITION
    dydt = x[1]         # x2 VELOCITY
    yi = x[2]           # x3

    u = Kp_c * (r - y) + Ki_c * yi - Kd_c * dydt         #PID output

    if abs(u)>force_constraint:
        print("A L E R T - GA Constraint")

    dxdt = [0,0,0]

    dxdt[0] = dydt
    dxdt[1] = (- c * dydt - k * y + u)/m
    dxdt[2] = r - y

    return [dxdt[0],dxdt[1],dxdt[2]]


solga_nc = solve_ivp(sys2PID_nc, [0, 20], x_ini, t_eval=tev)
y_out_nc = solga_nc.y[0, :]
t_out_nc = solga_nc.t
err_val_nc = 0.0
for y in range(len(t_out_nc)) :
    err_val_nc = err_val_nc + abs(set_interp(t_out_nc[y]) - y_out_nc[y])
print("Fitness value of GA without constraint: ", round(err_val_nc, 4))

solga_c = solve_ivp(sys2PID_c, [0, 20], x_ini, t_eval=tev)
y_out_c = solga_c.y[0, :]
t_out_c = solga_c.t
err_val_c = 0.0
for y in range(len(t_out_c)) :
    err_val_c = err_val_c + abs(set_interp(t_out_c[y]) - y_out_c[y])
print("Fitness value of GA with constraint: ", round(err_val_c, 4))

#Plotting code
plt.plot(tempo,set_point,"r--",label="Set Point Command [m]")
plt.plot(t_out_c, y_out_c, label="GA with Constraint Response [m]")
plt.plot(t_out_nc, y_out_nc, label="GA without Constraint Response [m]")
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude [m]')
# Set a title of the current axes.
plt.title('Setpoint for Constraint Comparison - Max setpoint = 80')
plt.xticks(np.arange(0, 20.5, step=1))
# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()