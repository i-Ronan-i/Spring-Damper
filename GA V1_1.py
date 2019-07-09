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
import time

def create_initial(pop_num, pop):
    """Creates the initial population of the genetic algorithm"""
    for s in range(pop_num):
        #Creating the random PID values
        kd_min, kd_max = 0, 100
        kp_min, kp_max = 0, 300
        ki_min, ki_max = 0, 300
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
    return pop

def fitness(MSDnum, MSDden, pop, times):
    """Calculates the fitness values of each member in pop[population]
    Also sets the simulation time in timesteps."""
    fit_val = []
    for s in range(len(pop)):
        #Create transfer functions for use by inputting current pop
        GAnum = [pop[s][0], pop[s][1], pop[s][2]]   # kd, kp, ki 
        GAden = [0, 1, 0]
        GHs_num = signal.convolve(GAnum, MSDnum)
        GHs_den = signal.convolve(GAden, MSDden)
        #Create CLTF
        cltf_num = GHs_num
        cltf_den = GHs_den + GHs_num
        cltf = signal.TransferFunction(cltf_num, cltf_den)
        
        #create transfer function with no poles or zeros
        unityF = signal.TransferFunction([1],[1])
        # Step functions
        t1, y1 = signal.step(unityF, T=times) #signal.step(errorsig)
        t2, y2 = signal.step(cltf, T=times)
        
        # This is a subtraction from the step function creating an
        # addition to the error value.
        err_vall = 0.0
        err_val = 0.0
        for o in range(len(times)):
            # repeats every s repetition
            err_vall = err_vall + abs(y1[o] - abs(y2[o]))
        err_val = err_vall * err_vall
        fit_val.insert(s, err_val)
    return fit_val
    
def crossover(a, b):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    new_a = []  #Clearing previous 
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut

    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]

    #Creates the new crossed-over list
    new_a = new_a1 + new_a2
    return new_a

def mutate(pop, mut_prob): 
    """Takes current population member and add a probability chance
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%."""
    pop_curr = pop
    for i in range(0, len(pop_curr)):
        for o in range(3) :
            if random.random() < mut_prob:
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.9, 2) #Maintains 2 d.p
                else :
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.1, 2) 
    return pop_curr

def create_next_generation(pop, pop_num, fit_val, mut_prob):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #This sorts the population into descending fitness
    switches = 1
    while switches > 0:
        switches = 0
        for i in range(len(fit_val)-1) :
            for j in range(i+1, len(fit_val)) : 
                if fit_val[i] > fit_val[j] :
                    temp = fit_val[i]
                    fit_val[i] = fit_val[j]
                    fit_val[j] = temp

                    temp2 = pop[i]
                    pop[i] = pop[j]
                    pop[j] = temp2

                    switches = switches + 1        
    #Pop list is now sorted. 

    #Next:
    #Saves top 1 performing genomes
    pop_top = []
    for m in range(1) :
        pop_top.append(pop[m])

    #Crossover performed in top 20
    pop_cross = []
    for n in range(20):
        new_pop1 = crossover(pop[n], pop[n+1])
        pop_cross.append(new_pop1)

    #Adds all currently available members
    #Then mutates them.
    pop_new = []
    pop_premut = []
    pop_premut = pop_top + pop_cross
    pop_new = mutate(pop_premut, mut_prob)

    #Create random members and saves
    for s in range(pop_num - len(pop_new)):
        #Creating the random PID values
        kd_min, kd_max = 0, 100
        kp_min, kp_max = 0, 300
        ki_min, ki_max = 0, 300
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop_new.append([kd_cur, kp_cur, ki_cur])
    return pop_new

def fit_sort(pop, fit_val):
    #This sorts the population into descending fitness (ascending order)
    switches = 1
    while switches > 0:
        switches = 0
        for i in range(len(fit_val)-1) :
            for j in range(i+1, len(fit_val)) : 
                if fit_val[i] > fit_val[j] :
                    temp = fit_val[i]
                    fit_val[i] = fit_val[j]
                    fit_val[j] = temp

                    temp2 = pop[i]
                    pop[i] = pop[j]
                    pop[j] = temp2

                    switches = switches + 1        
    #Pop list is now sorted. 
    return pop, fit_val

def main():
    start_time = time.time()
    #first thing is to create the system itself - MSD, input
    #MSD state definitions
    m = 2 # kg   - mass
    c = 10 # Ns/m - damping coefficient
    k = 20 # N/m  - spring coefficient

    # Transfer function definitions
    MSDnum = [0, 0, 1]
    MSDden = [m, c, k]
    pop_num = 40    #How large the initial population is
    pop = []
    mut_prob = 0.05  #probability for mutation set here
    timesteps = 8000 #Simulation length set here
    iteration_max = 30 #Total number of iterations and generations set here

    #Creating timestep table for step functions
    times=[]
    f=0.000
    n = 0.001
    for p in range(timesteps):
        times.append(f)
        f = f+n

    #Main GA call stack
    iteration = 0
    while iteration < iteration_max:
        if iteration == 0:
            pop = create_initial(pop_num, pop)
            fit_val = fitness(MSDnum, MSDden, pop, times)
            iteration = iteration + 1

        if iteration < iteration_max and iteration > 0:
            pop = create_next_generation(pop, pop_num, fit_val, mut_prob)
            fit_val = fitness(MSDnum, MSDden, pop, times)
            iteration = iteration + 1
    
    """This is the final section with the top solution being chosen and used"""
    #Final simulation run
    fit_val = fitness(MSDnum, MSDden, pop, times)
    pop, fit_val = fit_sort(pop, fit_val)
    print("Top overall Coefficients: kd = ", pop[0][0], "  kp = ", pop[0][1], "  ki = ", pop[0][2])
    print("Fitness value of top performing member: ", fit_val[0])

    #Create transfer functions for use by inputting current pop
    GAnum = [pop[0][0], pop[0][1], pop[0][2]]   # kd, kp, ki 
    GAden = [0, 1, 0]
    GHs_num = signal.convolve(GAnum, MSDnum)
    GHs_den = signal.convolve(GAden, MSDden)
    
    #Create CLTF
    cltf_num = GHs_num
    cltf_den = GHs_den + GHs_num
    cltf = signal.TransferFunction(cltf_num, cltf_den)

    #create transfer function with no poles or zeros
    unityF = signal.TransferFunction([1],[1])
    # Step functions
    t1, y1 = signal.step(unityF, T=times)
    t2, y2 = signal.step(cltf, T=times)

    #Creating the error signal
    y3 = []
    for g in range(timesteps):
        y3.append(y1[g] - y2[g])

    print("Time elapsed: ", time.time()-start_time)
    #Plotting graphs
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.title.set_text("Step Response")
    ax1.set_xlabel("Time(s)")
    ax1.set_ylabel("Amplitude (mm)")

    ax2.title.set_text("Error Signal")
    ax2.set_xlabel("Time(s)")
    ax2.set_ylabel("Error (mm)")

    ax2.plot(t1, y3, 'r', label='Error')
    ax1.plot(t2, y2, 'g', label='Amplitude (mm)')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.show()


    return True


main()
