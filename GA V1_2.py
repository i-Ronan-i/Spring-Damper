""" Attempting to create the GA PID controller for the MSD system """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt

def initialise():
    #first thing is to create the system itself - MSD, input
    #MSD state definitions
    m = 1 # kg   - mass
    c = 10 # Ns/m - damping coefficient
    k = 20 # N/m  - spring coefficient
    
    # PID initialising coefficients
    kd = 50 # derivative gain
    kp = 350 # proportional gain
    ki = 300 # integral gain

    # Transfer function definitions
    MSDnum = [0, 0, 1]
    MSDden = [m, c, k]
    GAnum = [kd, kp, ki]
    GAden = [0, 1, 0]
    GHs_num = signal.convolve(GAnum, MSDnum)
    GHs_den = signal.convolve(GAden, MSDden)

    cltf_num = GHs_num
    cltf_den = GHs_den + GHs_num
    cltf = signal.TransferFunction(cltf_num, cltf_den)
    #Error signal is input - output = input(1 - GA*MSD)
    error_num = GHs_den - GHs_num
    error_den = GHs_den
    errorsig = signal.TransferFunction(error_num, error_den)
    print(errorsig)
    
    #Creating timestep table for step functions
    times=[]
    stepplot=[]
    f=0.000
    n = 0.001
    timesteps = 2000
    for s in range(timesteps):
        stepplot.append(1)
        times.append(f)
        f = f+n

    # Step function in use
    t1, y1 = 0,0 #signal.step(errorsig)
    t2, y2 = signal.step(cltf, T=timesteps)

    # this is a workaround for calculating the error value without 
    # the proper step input function being used for error signal values.
    """ err = []
    for s in range(timesteps):
        err.append(stepplot[s] - y2[s])

    for s in range(timesteps-1):
        err_val = err[s] + err[s+1]
    err_val = err_val*err_val
    
    print("Error value = ", err_val)
    """
    #Plotting graphs
    fig, ax1 = pylab.subplots()
    ax1.plot(t1, y1,'g', linewidth=1.5)
    ax1.legend()
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('Error',color='g')
    pylab.title('Plot')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude',color='b')
    ax2.plot(t2, y2,'b', label = r'$x (mm)$', linewidth=1)
    pylab.grid
    pylab.show()



member = 0

def fitness(member):
    return 1
    
def crossover(a, b):
    return a

def mutate(member): 
    return member

def create_new_member():
    return member

def create_next_generation(population):
    return population

def main(number_of_iterations):
    return True

""" Creating a random number

"""
initialise()
