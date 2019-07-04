
import numpy as np
from numpy import array
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt

#first thing is to create the system itself - MSD, input
#MSD state definitions
m = 4.5 # kg   - mass
c = 0.8 # Ns/m - damping coefficient
k = 100 # N/m  - spring coefficient

# PID initialising coefficients
kd = 0.2 # derivative gain
kp = 0.2 # proportional gain
ki = 0.2 # integral gain

MSDnum = [0, 0, 1]
MSDden = [m, c, k]
GAnum = [kd, kp, ki]
GAden = [0, 1, 0]
GHs_num = signal.convolve(GAnum, MSDnum)
GHs_den = signal.convolve(GAden, MSDden)

GHs = signal.TransferFunction(GHs_num, GHs_den)


Xs = GHs

t, y= signal.step2(GHs)
#t2, y2 = signal.step2(errorsig)

#Plotting graphs
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude (mm)')
plt.title('Step response for MSD System')
plt.grid()
plt.show()