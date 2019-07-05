""" Attempting to create the GA PID controller for the MSD system """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random

def initialise(cltf, ):
    #first thing is to create the system itself - MSD, input
    #MSD state definitions
    m = 2 # kg   - mass
    c = 10 # Ns/m - damping coefficient
    k = 20 # N/m  - spring coefficient
    
    # PID initialising coefficients
    kd = 0.0 # derivative gain
    kp = 0.0 # proportional gain
    ki = 0.0 # integral gain

    # Transfer function definitions
    MSDnum = [0, 0, 1]
    MSDden = [m, c, k]
    GAnum = [kd, kp, ki]
    GAden = [0, 1, 0]
    GHs_num = signal.convolve(GAnum, MSDnum)
    GHs_den = signal.convolve(GAden, MSDden)
    print("GHs_num: ", GHs_num)
    print("GHs_den: ", GHs_den)

    cltf_num = GHs_num
    cltf_den = GHs_den + GHs_num
    cltf = signal.TransferFunction(cltf_num, cltf_den)
    #Error signal is input - output = input(1 - GA*MSD)
    """error_num = GHs_den - GHs_num
    error_den = GHs_den
    errorsig = signal.TransferFunction(error_num, error_den)
    print("Error_num: ", error_num)
    print("error_den: ", error_den)
    print("errorsig: ", errorsig) """
    
    #Creating timestep table for step functions
    times=[]
    stepplot=[]
    f=0.000
    n = 0.001
    timesteps = 3000
    for s in range(timesteps):
        stepplot.append(1)
        times.append(f)
        f = f+n

    unityF = signal.TransferFunction([1],[1])
    # Step function in use
    t1, y1 = signal.step(unityF, T=times) #signal.step(errorsig)
    t2, y2 = signal.step(cltf, T=times)
    
    # This manipulates the unity step input function to produce the error plot via
    # subtraction from the step function and also creates the error value.
    errY = []
    err_val = 0
    for s in range(timesteps):
        errY.append(y1[s] - y2[s])
        err_val = err_val + errY[s]
    err_valsq = err_val*err_val
    print("Fitness Error value = ", err_valsq, "    Error value: ", err_val)
    
    #Plotting graphs
    fig, ax1 = pylab.subplots()
    ax1.plot(t1, errY,'g', linewidth=1.5)
    ax1.legend()
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('Error',color='g')
    pylab.title('Step Response and Error Signal')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Amplitude',color='b')
    ax2.plot(t2, y2,'b', label = r'$x (mm)$', linewidth=1)
    ax1.axis([0, 3, 0, 1.2])
    ax2.axis([0, 3, 0, 1.2])
    pylab.grid()
    pylab.show()



member = 0

def fitness(member):
    return 1
    
def crossover(a, b):
    return a

def mutate(member): 
    return member

def create_new_member():
    population = 40
    pop = []
    for s in range(population):
        #Creating the random PID values
        kd_min, kd_max = 0, 100
        kp_min, kp_max = 0, 300
        ki_min, ki_max = 0, 250
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
        print("Current Population Member: ", pop[s])
    return member

def create_next_generation(population):
    return population

def main(number_of_iterations):
    return True

#initialise()
create_new_member()