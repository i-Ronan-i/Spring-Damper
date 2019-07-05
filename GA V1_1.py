""" Attempting to create the GA PID controller for the MSD system """
#Created initial population

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random

def initialise():
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




    
    """Plotting graphs
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
    pylab.show()"""
    return MSDnum, MSDden

member = 0

def create_new_member():
    """Creates the initial population of the genetic algorithm"""
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
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
        print("Current Population Member: ", pop[s])
    return population, pop


def fitness(MSDnum, MSDden, population, pop):
    """Calculates the fitness values of each member in pop[population]"""
    for s in range(population):
        #Create transfer functions for use by inputting current pop
        GAnum = [pop[s][1], pop[s][2], pop[s][3]]
        GAden = [0, 1, 0]
        GHs_num = signal.convolve(GAnum, MSDnum)
        GHs_den = signal.convolve(GAden, MSDden)
        #Create CLTF
        cltf_num = GHs_num
        cltf_den = GHs_den + GHs_num
        cltf = signal.TransferFunction(cltf_num, cltf_den)
        
        #Creating timestep table for step functions
        times=[]
        f=0.000
        n = 0.001
        timesteps = 3000
        for m in range(timesteps):
            times.append(f)
            f = f+n

        #create transfer function with no poles or zeros
        unityF = signal.TransferFunction([1],[1])
        # Step functions
        t1, y1 = signal.step(unityF, T=times) #signal.step(errorsig)
        t2, y2 = signal.step(cltf, T=times)
        
        # This is a subtraction from the step function creating an
        # addition to the error value.
        err_val[s] = [0]
        fit_val = []
        for o in range(timesteps):
            err_val[s] = err_val[s] + y1[o] - y2[o]
            fit_val[s] = err_val[s]*err_val[s]
            print("Fitness Pop ", s, " error value = ", fit_val)
    for m in range(population):
        print("Fitness value for Pop ", s, " is :", fit_val[m])
    return 1
    
def crossover(a, b):
    return a

def mutate(member): 
    return member

def create_next_generation(population):
    return population

def main(number_of_iterations):
    return True

initialise()
create_new_member()
fitness(MSDnum, MSDden, population, pop)