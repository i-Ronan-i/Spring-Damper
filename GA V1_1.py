""" Attempting to create the GA PID controller for the MSD system """
#Created initial population
#Fitness function working now
#Mutate and Crossover operations complete
#Sorting via fitness complete

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random

def create_initial(pop_num, pop):
    """Creates the initial population of the genetic algorithm"""
    for s in range(pop_num):
        #Creating the random PID values
        kd_min, kd_max = 0, 100
        kp_min, kp_max = 0, 300
        ki_min, ki_max = 0, 250
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
    return pop


def fitness(MSDnum, MSDden, pop_num, pop):
    """Calculates the fitness values of each member in pop[population]
    Also sets the simulation time in timesteps."""
    fit_val = []
    for s in range(pop_num):
        #Create transfer functions for use by inputting current pop
        GAnum = [pop[s][0], pop[s][1], pop[s][2]]   # kd, kp, ki 
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
        for p in range(timesteps):
            times.append(f)
            f = f+n

        #create transfer function with no poles or zeros
        unityF = signal.TransferFunction([1],[1])
        # Step functions
        t1, y1 = signal.step(unityF, T=times) #signal.step(errorsig)
        t2, y2 = signal.step(cltf, T=times)
        
        # This is a subtraction from the step function creating an
        # addition to the error value.
        err_vall = 0.0
        err_val = 0.0
        for o in range(timesteps):
            # repeats every s repetition
            err_vall = err_vall + y1[o] - y2[o]
        err_val = err_vall * err_vall
        fit_val.insert(s, err_val)
    for m in range(pop_num):
        print("Fitness value for Pop Member ", m, " is :", fit_val[m])
    return fit_val
    
def crossover(a, b):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    print("a: ", a)
    print("b: ", b)
    new_a = []  #Clearing previous 
    new_b = []  #list values
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut
    cut_b = random.randint(1, len(b)-1) #Can only be at position 1 or 2.
    print("Cut_a: ", cut_a, "    Cut_b: ", cut_b)

    
    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]
    print("New_a1: ", new_a1)
    print("New_a2: ", new_a2)

    new_b1 = b[0 : cut_b]
    new_b2 = a[cut_b : len(a)]
    print("New_b1: ", new_b1) 
    print("New_b2: ", new_b2)     

    #Creates the new crossed-over lists
    new_a = new_a1 + new_a2
    new_b = new_b1 + new_b2
    print("New_a: ", new_a, "     New_b: ", new_b)
    return new_a, new_b

def mutate(pop): 
    """Takes current population member and add a probability chance
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%."""
    mut_prob = 0.03
    mut_pop = pop
    for i in range(0, len(pop_curr)):
        print("pop_curr[i]: ", pop_curr[i])
        if random.random() < mut_prob:
            if random.random() < 0.5:
                pop_curr[i] = round(pop_curr[i] * 0.9, 2) #Maintains 2 d.p
            else :
                pop_curr[i] = round(pop_curr[i] * 1.1, 2) 
        print("pop_curr[i]: ", pop_curr[i])
    return pop_curr

def create_next_generation(pop, pop_num, fit_val, mut_prob):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #This sorts the population into descending fitness
    switches = 1
    while switches > 0:
        switches = 0
        print("switches: ", switches)
        for i in range(len(fit_val)-1) :
            for j in range(i+1, len(fit_val)) : 
                if fit_val[i] > fit_val[j] :
                    temp = fit_val[i]
                    print("fit_val1[i]: ", fit_val[i])
                    fit_val[i] = fit_val[j]
                    fit_val[j] = temp
                    print("fit_val1[i]: ", fit_val[i])

                    temp2 = pop[i]
                    print("pop1[i]: ", pop[i])
                    pop[i] = pop[j]
                    pop[j] = temp2
                    print("pop1[i]: ", pop[i])

                    switches = switches + 1        
    #Pop list is now sorted. 

    #Next:
    #Save top 5 aside
    #Use crossover in top 20 and save over initial 20 - will require another loop that sends pop[i] & pop[i+1] till pop[19]
    #Add 5 back in, mutate 25 of pop
    #Create 15 random and save in pop[25] -> pop[40]



    return pop_new

def main():
    #first thing is to create the system itself - MSD, input
    #MSD state definitions
    m = 2 # kg   - mass
    c = 10 # Ns/m - damping coefficient
    k = 20 # N/m  - spring coefficient

    # Transfer function definitions
    MSDnum = [0, 0, 1]
    MSDden = [m, c, k]
    pop_num = 40
    pop = []

    iteration = 0
    while iteration < 100:
        print(iteration)
        if iteration == 0:
            pop = create_initial(pop_num, pop)
            fit_val = fitness(MSDnum, MSDden, pop_num, pop)
            iteration = iteration + 1

        if iteration < 100:
            pop = create_next_generation(pop, pop_num, fit_val, mut_prob)
            fit_val = fitness(MSDnum, MSDden, pop_num, pop)
            iteration = iteration + 1
        else:
            winning_pop()
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

    return True

def winning_pop():
    return 1


#mut_prob = 0.03 #3% chance of mutation
#mutate(pop_curr, mut_prob)
