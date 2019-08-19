from numpy import *
from scipy.integrate import solve_ivp, RK23, odeint
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np
from operator import *
def Sqrt(a):
    if a > 0:
        return np.sqrt(a)
    else:
        return np.abs(a)


def Log(a):
    if a > 0:
        return np.log(a)
    else:
        return np.abs(a)


def Tanh(a):
    return np.tanh(a)

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

tempo = np.linspace(0, 20, 20*5)
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))
#"""
"""
tempo = np.linspace(0, 20, 20*10)
set_point = np.sin(tempo)
#"""
set_interp = PchipInterpolator(tempo, set_point)

a0 = 1  # M  [Kg]
a1 = 10  # C  [Ns/m]
a2 = 20  # K  [N/m]
rise_time = 0.1
r_max = np.amax(set_point)
r_min = np.amin(set_point)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max
force_constraint = a1 * vr + a2 * yr + a0 * ar
print("Force Constraint: ", force_constraint)

def sys2PID(t,x):
    global force_constraint
    Kp = 452.59  # Proportional Gain
    Ki = 416.12 # Integrative Gain
    Kd = 14.76 # Derivative Gain
    #Time to complete: 113.72mins
    #Fitness: 46.3401
    #SinFitness: 32.8507
    #Sin Force Constraint 417.95 - No Breaches

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

def sys2GP(t,x):
    global force_constraint
    r=set_interp(t)
    #Fitness: 30.0561
    #SinFitness: 66.4221
    #Sin Force Constraint 417.95 - No Breaches
    #State Variables
    y = x[0]            # x1 POSITION
    d_err = x[1]         # x2 VELOCITY
    i_err = x[2]           # x3

    err=r-y

    u=multiply(multiply(absolute(multiply(i_err, 63.6844)), 63.6844), multiply(multiply(multiply(i_err, 63.6844), multiply(Tanh(add(48.2562, Sqrt(pi))), i_err)), multiply(Tanh(Log(5)), err)))

    if abs(u)>force_constraint:
        print("A L L E R T_GP")

    dxdt = [0,0,0]

    dxdt[0] = d_err
    dxdt[1] = (- a1 * d_err - a2 * y + u)/a0
    dxdt[2] = err

    return [dxdt[0],dxdt[1],dxdt[2]]

#changed from np.linspace just because I didn't like the lack of rounding with each point
temp = round(0.00, 2)
dt=0.02
tev = []
for times in range(int(20/dt)):
    tev.append(temp)
    temp = round(temp + dt, 2)
x_ini = [0,0,0]        # initial conditions
solga = solve_ivp(sys2PID, [0, 20], x_ini, t_eval=tev)
yga = solga.y[0, :]
tga = solga.t

err_val = 0.0
for y in range(len(tga)) :
    err_val = err_val + abs(set_interp(tga[y]) - yga[y])
print("Fitness value of GA Tuned PID: ", round(err_val, 4))

solgp = solve_ivp(sys2GP, [0, 20], x_ini, t_eval=tev)
ygp = solgp.y[0, :]
tgp = solgp.t
err_val_gp = 0.0
for y in range(len(tgp)) :
    err_val_gp = err_val_gp + abs(set_interp(tgp[y]) - ygp[y])
print("Fitness value of GP controller: ", round(err_val_gp, 4))

plt.title("Response Comparison Between GA Tuned PID Controller and GP Controller")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [m]")
plt.plot(tga,yga,label="Mass Position with GA controller [m]")
plt.plot(tgp,ygp,label="Mass Position with GP controller [m]")
plt.plot(tempo,set_point,"r--",label="Setpoint Command [m]")
plt.xticks(np.arange(0, 20.5, step=1))
plt.legend()
plt.grid()
plt.show()