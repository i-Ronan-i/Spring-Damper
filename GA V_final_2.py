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

def create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times):
    """Creates the initial population of the genetic algorithm while making sure it adheres to force constraints"""
    for s in range(pop_num):
        flag = True
        while flag == True:
            flagged = 0  
            #Creating the random PID values
            kd_cur = round(random.uniform(kd_min, kd_max), 2)
            kp_cur = round(random.uniform(kp_min, kp_max), 2)
            ki_cur = round(random.uniform(ki_min, ki_max), 2) 

            #Simulate for PID values
            GAnum = [kd_cur, kp_cur, ki_cur]   # kd, kp, ki 
            GAden = [0, 1, 0]
            GHs_num = signal.convolve(GAnum, MSDnum)
            GHs_den = signal.convolve(GAden, MSDden)
            #Create CLTF
            cltf_num = GHs_num
            cltf_den = GHs_den + GHs_num
            cltf = signal.TransferFunction(cltf_num, cltf_den)
    
            #Simulates the current system [s].
            t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)        
            err_sig = []
            for y in range(len(times)) :
                err_sig.append(desired_x[y] - y_out[y])
                
            for y in range(1, len(err_sig)) :
                y_curr = []
                y_curr = [err_sig[y-1], err_sig[y]]
                err_int = np.trapz(y_curr, dx=dt)
                err_diff = (y_curr[1]-y_curr[0])/dt
        
                #pid force =   kp * e              ki * i_e         kd * d_e
                pid_force = kp_cur*err_sig[y] + ki_cur*err_int + kd_cur*err_diff
                #if PID values exceeds requirements for time step, flag it.
                if system_force < pid_force:
                    flagged = flagged + 1

            #If system never exceeds max force then continue.
            if flagged == 0:
                flag = False
        
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

        #sets the tolerance time range for error in the last 4 time steps
        for y in range(len(times)) :
            err_val = err_val + abs(desired_x[y] - y_out[y]) #**2 #- Removing the square as it converges to quick settling time but doesn't reach set points     
        
        fit_val.insert(s, err_val)
    return fit_val
    
def crossover(a, b, system_force, desired_x, times, dt, MSDnum, MSDden):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    new_a = []  #Clearing previous 
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut

    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]

    #Creates the new crossed-over list
    new_a = new_a1 + new_a2
    #trying to keep force constraints upheld
    
    #Simulate for PID values
    GAnum = [new_a[0], new_a[1], new_a[2]]   # kd, kp, ki 
    GAden = [0, 1, 0]
    GHs_num = signal.convolve(GAnum, MSDnum)
    GHs_den = signal.convolve(GAden, MSDden)
    #Create CLTF
    cltf_num = GHs_num
    cltf_den = GHs_den + GHs_num
    cltf = signal.TransferFunction(cltf_num, cltf_den)

    #Simulates the current system [s].
    t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)        
    err_sig = []
    for y in range(len(times)) :
        err_sig.append(desired_x[y] - y_out[y])
    
    flagged = 0
    for y in range(1, len(err_sig)) :
        y_curr = []
        y_curr = [err_sig[y-1], err_sig[y]]
        err_int = np.trapz(y_curr, dx=dt)
        err_diff = (y_curr[1]-y_curr[0])/dt

        #pid force =     kp * e                ki * i_e           kd * d_e
        pid_force = new_a[1]*err_sig[y] + new_a[2]*err_int + new_a[0]*err_diff
        #if PID values exceeds requirements for time step, flag it.
        if system_force < pid_force:
            flagged = flagged + 1

    #If system exceeds max force then crossover is aborted.
    if flagged > 0:
        new_a = a
    return new_a

def mutate(pop, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times) :
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
                        
        #Trying to keep force contraints upheld
        flag = True
        kpbig = 0
        kibig = 0
        kdbig = 0
        while flag == True:
            flagged = 0  
            #Simulate for PID values
            GAnum = [pop[i][0], pop[i][1], pop[i][2]]   # kd, kp, ki 
            GAden = [0, 1, 0]
            GHs_num = signal.convolve(GAnum, MSDnum)
            GHs_den = signal.convolve(GAden, MSDden)
            #Create CLTF
            cltf_num = GHs_num
            cltf_den = GHs_den + GHs_num
            cltf = signal.TransferFunction(cltf_num, cltf_den)
    
            #Simulates the current system [s].
            t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)        
            err_sig = []
            for y in range(len(times)) :
                err_sig.append(desired_x[y] - y_out[y])
            
            flagged = 0
            for y in range(1, len(err_sig)) :
                y_curr = []
                y_curr = [err_sig[y-1], err_sig[y]]
                err_int = np.trapz(y_curr, dx=dt)
                err_diff = (y_curr[1]-y_curr[0])/dt
        
                #pid force = kp * e +  ki * i_e + kd * d_e
                kp_force = pop[i][1]*err_sig[y]
                ki_force = pop[i][2]*err_int
                kd_force = pop[i][0]*err_diff
                pid_force = kp_force + ki_force + kd_force
                #counters for mutation reduction if force is exceeded
                if kp_force > ki_force and kp_force > kd_force :
                    kpbig = kpbig+1
                if ki_force > kp_force and ki_force > kd_force :
                    kibig = kibig+1
                if kd_force > ki_force and kd_force > kp_force :
                    kdbig = kdbig+1
                #if PID values exceeds requirements for time step, flag it.
                if system_force < pid_force:
                    flagged = flagged + 1
            #If system never exceeds max force then continue.
            if flagged == 0:
                flag = False
            else :
                #Mutation reduction of biggest contributing factor
                if kpbig > kibig and kpbig > kdbig:
                    pop_curr[i][1] = pop_curr[i][1] * 0.99
                if kibig > kpbig and kibig > kdbig:
                    pop_curr[i][2] = pop_curr[i][2] * 0.99
                if kdbig > kibig and kdbig > kpbig:
                    pop_curr[i][0] = pop_curr[i][0] * 0.99
    return pop_curr

def create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #Saves top 3 performing genomes
    pop_top = []
    for m in range(1) :
        pop_top.append(pop[m])

    #Crossover performed in top 20
    pop_cross = []
    for n in range(25):
        new_pop1 = crossover(pop[n], pop[n+1], system_force, desired_x, times, dt, MSDnum, MSDden)
        pop_cross.append(new_pop1)

    #Adds all currently available members
    #Then mutates them.
    pop_new = []
    pop_premut = []
    pop_premut = pop_top + pop_cross
    pop_new = mutate(pop_premut, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times)

    #Create random members and saves   
    for s in range(pop_num - len(pop_new)):
        flag = True
        while flag == True:
            flagged = 0  
            #Creating the random PID values
            kd_cur = round(random.uniform(kd_min, kd_max), 2)
            kp_cur = round(random.uniform(kp_min, kp_max), 2)
            ki_cur = round(random.uniform(ki_min, ki_max), 2) 

            #Simulate for PID values
            GAnum = [kd_cur, kp_cur, ki_cur]   # kd, kp, ki 
            GAden = [0, 1, 0]
            GHs_num = signal.convolve(GAnum, MSDnum)
            GHs_den = signal.convolve(GAden, MSDden)
            #Create CLTF
            cltf_num = GHs_num
            cltf_den = GHs_den + GHs_num
            cltf = signal.TransferFunction(cltf_num, cltf_den)
    
            #Simulates the current system [s].
            t_out, y_out, state = signal.lsim(cltf, U=desired_x, T=times)        
            err_sig = []
            for y in range(len(times)) :
                err_sig.append(desired_x[y] - y_out[y])
                
            for y in range(1, len(err_sig)) :
                y_curr = []
                y_curr = [err_sig[y-1], err_sig[y]]
                err_int = np.trapz(y_curr, dx=dt)
                err_diff = (y_curr[1]-y_curr[0])/dt
        
                #pid force =   kp * e              ki * i_e         kd * d_e
                pid_force = kp_cur*err_sig[y] + ki_cur*err_int + kd_cur*err_diff
                #if PID values exceeds requirements for time step, flag it.
                if system_force < pid_force:
                    flagged = flagged + 1
            #If system never exceeds max force then continue.
            if flagged == 0:
                flag = False
        
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
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


"""""""""""""""""""""""""""""""""---------- MAIN CODE PROGRAM BODY ----------------"""""""""""""""""""""""""""""""""""""""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

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
dt = 0.01
timesteps = 20/dt #Simulation length set here
iteration_max = 200 #Total number of iterations and generations set here
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
desired_x   = np.sin(timearray)
#"""

#Creating timestep table for step functions
times=[]
f=0.00
for p in range(int(timesteps)):
    times.append(f)
    f = f+dt

#System force contraint
system_force = round(m*(2*max(desired_x)/0.1) + c*(0.9*max(desired_x)) + k*(2*max(desired_x)/(0.1**2)), 0)

#Main GA call stack
iteration = 0
while iteration < iteration_max:
    if iteration == 0:
        pop = create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times)
        fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
        iteration = iteration + 1

    if iteration < iteration_max and iteration > 0:
        pop, fit_val = fit_sort(pop, fit_val)
        pop = create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max, desired_x, system_force, dt, MSDnum, MSDden, times)
        fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
        iteration = iteration + 1

"""This is the final section with the top solution being chosen and used"""
#Final simulation run
fit_val = fitness(MSDnum, MSDden, pop, times, desired_x)
pop, fit_val = fit_sort(pop, fit_val)
print("Top overall Coefficients: kd = ", pop[0][0], "  kp = ", pop[0][1], "  ki = ", pop[0][2])
print("Fitness value of top performing member: ", round(fit_val[0], 4))
print("Time elapsed: ", round((time.time()-start_time)/60, 2), " minutes")

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

plt.xticks(np.arange(0, 20.5, step=1))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()






"""
while flag = True :
    flag = False
    
    #Create PID values
    #Simulate for PID values

    err_sig = []

    for y in range(len(times)) :
        err_sig.append(desired_x[y] - y_out[y])
      
    for y in range(1, len(err_sig)) :
            y_curr = []
            y_curr = [err_sig[y-1], err_sig[y]
            err_int = (np.trapz(y_curr, dt))
            err_diff = (np.gradient(y_curr, dt))

        #pid force =      kp * e                 ki * i_e                 kd * d_e
        pid_force = pop[s][1]*err_sig[y] + pop[s][2]*err_int + pop[s][0]*err_diff
        if system_force < pid_force:
            flag = True
"""