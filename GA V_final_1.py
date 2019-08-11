""" Genetic Algorithm Tuning of a PID controller 
for offline optimisation of a Mass-Spring-Damper System. """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time

def create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force):
    """Creates the initial population of the genetic algorithm while making sure it adheres to force constraints"""

    for s in range(pop_num):
            #Creating the random PID values
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
    return pop

def fitness(MSDnum, MSDden, pop, times, desired_x):
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

        #Simulates the current system [s].
        t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)

        err_val = 0.0
        err_vall = 0.0

        #sets the tolerance time range for error in the last 4 time steps
        for y in range(len(times)) :
            err_vall = err_vall + abs(desired_x[y] - y_out[y]) #**2 - Removing the square as it converges to quick settling time but doesn't reach set points
        err_val = err_vall 

        #topval = pop[s][0] + pop[s][1] + pop[s][2]

        #fitval = 0.0001*topval + 0.9999*err_val

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

def mutate(pop, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max): 
    """Takes current population member and add a probability chance
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%."""
    pop_curr = pop
    for i in range(0, len(pop_curr)):
        for o in range(3) :
            if random.random() < mut_prob:
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.95, 2) #Maintains 2 d.p
                else :
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.05, 2)
                    if pop_curr[i][0] > kd_max :
                        pop_curr[i][0] = float(kd_max) 
                    if pop_curr[i][1] > kp_max :
                        pop_curr[i][1] = float(kp_max)
                    if pop_curr[i][2] > ki_max :
                        pop_curr[i][2] = float(ki_max)
    return pop_curr

def create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #Saves top 3 performing genomes
    pop_top = []
    for m in range(1) :
        pop_top.append(pop[m])

    #Crossover performed in top 20
    pop_cross = []
    for n in range(25):
        new_pop1 = crossover(pop[n], pop[n+1])
        pop_cross.append(new_pop1)

    #Adds all currently available members
    #Then mutates them.
    pop_new = []
    pop_premut = []
    pop_premut = pop_top + pop_cross
    pop_new = mutate(pop_premut, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)

    #Create random members and saves
    flag = False
    
    for s in range(pop_num - len(pop_new)):
        #Creating the random PID values
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


""" MAIN CODE PROGRAM BODY """
start_time = time.time()
#first thing is to create the system itself - MSD, input
#MSD state definitions
m = 1 # kg   - mass
c = 10 # Ns/m - damping coefficient
k = 20 # N/m  - spring coefficient

# Transfer function definitions
#step_mag = 1    #Magnitude of the step function - manupulation of functions, not actual step function
MSDnum = [0, 0, 1]
MSDden = [m, c, k]
pop_num = 60    #How large the initial population is
pop = []
mut_prob = 0.08  #probability for mutation set here
timesteps = 20/0.01 #Simulation length set here
iteration_max = 120 #Total number of iterations and generations set here
desired_x = []
#Minimum and maximum PID coefficient gains.
kd_min, kd_max = 0, 500
kp_min, kp_max = 0, 500
ki_min, ki_max = 0, 900

#Create the desired output plot for lsim
#"""
#20s for creating the amplitude values for the desired values of the system
for b in range(1) :
#5s
    for a in range(400) :
        desired_x.append(2)
    for a in range(500) :
        desired_x.append(4)
    for a in range(500) :
        desired_x.append(8)
    for a in range(600) :
        desired_x.append(3)
    #for a in range(100) :
    #    desired_x.append(1)

#"""
"""
# Get x values of the sine wave
timearray        = np.arange(0, 20, 0.01);
# Amplitude of the sine wave is sine of a variable like time
desired_x   = 2*np.sin(2*timearray)
#"""

#Creating timestep table for step functions
times=[]
f=0.00
n = 0.01 #100Hz
for p in range(int(timesteps)):
    times.append(f)
    f = f+n


#Main GA call stack
iteration = 0
while iteration < iteration_max:
    if iteration == 0:
        pop = create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force)
        fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
        iteration = iteration + 1

    if iteration < iteration_max and iteration > 0:
        pop, fit_val = fit_sort(pop, fit_val)
        pop = create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force)
        fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
        iteration = iteration + 1

"""This is the final section with the top solution being chosen and used"""
#Final simulation run
fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
pop, fit_val = fit_sort(pop, fit_val)
print("Top overall Coefficients: kd = ", pop[0][0], "  kp = ", pop[0][1], "  ki = ", pop[0][2])
print("Fitness value of top performing member: ", round(fit_val[0], 4))
print("Time elapsed: ", time.time()-start_time)

#Create transfer functions for use by inputting current pop
GAnum = [pop[0][0], pop[0][1], pop[0][2]]   # kd, kp, ki 
GAden = [0, 1, 0]
GHs_num = signal.convolve(GAnum, MSDnum)
GHs_den = signal.convolve(GAden, MSDden)

#Create CLTF
cltf_num = GHs_num
cltf_den = GHs_den + GHs_num
cltf = signal.TransferFunction(cltf_num, cltf_den)

#Simulates the current system [s].
t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)

#Plotting code
plt.plot(times, desired_x, label = "Desired Output - X_d(s)")
plt.plot(times, y_out, label = "System Output - X(s)") 
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude')
# Set a title of the current axes.
#plt.title('System Response to Varying Step Inputs')
plt.title('System Response over 20 Seconds.')

plt.xticks(np.arange(0, 20.5, step=2))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()