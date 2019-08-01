""" Using GA V2_2 code as foundation and changing the desired output to a repeated pattern """

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time


def Create_initial_generation(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Creates the initial population of the genetic algorithm"""
    for s in range(pop_num):
        #Creating the random PID values
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kd_cur, kp_cur, ki_cur])
    return pop


def Create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #Saves top 1 performing genomes
    pop_top = []
    for m in range(5) :
        pop_top.append(pop[m])

    #Crossover performed in top 20
    pop_cross = []
    for n in range(30):
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
        kd_cur = round(random.uniform(kd_min, kd_max), 2)
        kp_cur = round(random.uniform(kp_min, kp_max), 2)
        ki_cur = round(random.uniform(ki_min, ki_max), 2)
        #Into 2-D List. Access via pop[i][j]
        pop_new.append([kd_cur, kp_cur, ki_cur])
    return pop_new


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
        for o in range(len(pop_curr[i])) :
            if random.random() < mut_prob:
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.9, 2) #Maintains 2 d.p
                else :
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.1, 2) 
    return pop_curr


def Fit_sort(pop, fit_val):
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


def Fitness(MSDnum, MSDden, pop, time_steps, desired_x, y_global, logged_time):
    """Calculates the fitness values of each member in pop[population]"""
    fit_val = []
    desired_x_curr = []
    #Creates array of desired_x for the current timesteps - Tries to follow the current output plot then adds the desired_x values at  the end 
    #for b in range(len(logged_time[changes])) :
    #    desired_x_curr.append(y_global[b])
    #for b in range(len(logged_time[changes]), len(time_steps)) :
    #    desired_x_curr.append(desired_x[b]) 
    for b in range(len(time_steps)) :
        desired_x_curr.append(desired_x[b])
    
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

        t_out, y_out, state = signal.lsim(cltf, U=desired_x_curr, T=time_steps)

        err_vall = 0.0
        err_val = 0.0
        #CURRENTLY THE WAY TO DETECT THE LAST 4 REAL Y OUTPUTS ON THE GLOBAL
        #Y OUTPUT. Y TEMP = Y GLOBAL + Y OUT IN LATEST STEPS MADE SINCE LAST CHANGE
        y_temp = []
        y_temp = y_global.copy()
        for y in range(len(logged_time[changes-1]), len(time_steps)) : #changes - 1 because it increments just before this part of the code
            y_temp.insert(y, y_out[y]) 
        #sets the tolerance for error in the last 4 time steps
        for y in range(len(time_steps)-4, len(time_steps)) :
            err_vall = err_vall + abs(desired_x_curr[y]) - abs(y_temp[y])    
        fit_val.insert(s, err_vall * err_vall)
  
    return fit_val


def Recognise(changes, y_global, time_curr, log_time, time_steps, MSDnum, MSDden, pop, desired_x) : 
    """Function for recognising if a change has occured to the system or if
    improvement is needed in the performance.
    Runs its own version of fitness detection"""
    recognise = False
    y_temp = []
    desired_x_curr = []
    #Create transfer functions for use by inputting current pop
    GAnum = [pop[0][0], pop[0][1], pop[0][2]]   # kd, kp, ki 
    GAden = [0, 1, 0]
    GHs_num = signal.convolve(GAnum, MSDnum)
    GHs_den = signal.convolve(GAden, MSDden)
    #Create CLTF
    cltf_num = GHs_num
    cltf_den = GHs_den + GHs_num
    cltf = signal.TransferFunction(cltf_num, cltf_den)

    #Creates array of desired_x for the current timesteps
    #This starts with the previous signal path then adds the next desired x values to aim for
    for b in range(len(logged_time[changes])) :
        desired_x_curr.append(y_global[b])
    for b in range(len(logged_time[changes]), len(time_steps)) :
        desired_x_curr.append(desired_x[b]) 
   
    #Simulates the current top performing system.
    t_out, y_out, state = signal.lsim(cltf, U=desired_x_curr, T=time_steps)

    rec_val = 0.0
    err_vall = 0.0
    #CURRENTLY THE WAY TO DETECT THE LAST 4 REAL Y OUTPUTS ON THE GLOBAL Y OUTPUT. 
    #Y TEMP = Y GLOBAL + Y OUT( FOR LATEST STEPS MADE SINCE LAST CHANGE)
    y_temp = y_global.copy()
    for y in range(len(logged_time[changes]), len(time_steps)) :
        y_temp.insert(y, y_out[y]) 

    #sets the tolerance time range for error in the last 4 time steps
    for y in range(len(time_steps)-4, len(time_steps)) :
        err_vall = err_vall + abs(desired_x_curr[y]) - abs(y_temp[y])   
    rec_val = err_vall * err_vall
    
    #This is setting the recognition level tolerance for changing the controller
    #and the time between allowed controller reactions.
    if rec_val > 0.000 and time_curr-log_time >= 0.08:
        y_global = y_temp.copy()       
        changes = changes + 1
        logged_time.append(time_steps.copy())
        recognise = True
    
    #for the end of program run to get the final portion of the system output.
    if time_curr == 10 :
        y_global = y_temp.copy()
    return recognise, changes, y_global, logged_time


def GA_controller(time_steps, pop, fit_val):
    """Creates the generations, fitness has been moved to another function"""
    pop_num = 80
    mut_prob = 0.03
    kd_min, kd_max = 0, 100
    kp_min, kp_max = 0, 400
    ki_min, ki_max = 0, 450

    if len(time_steps) == 1 :
        pop = Create_initial_generation(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
    else :
        pop = Create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
    return pop


def initialise():
    """Sets desired_x inputs - currently for 10 seconds
    and other global variables. """
    desired_x = []

    #"""
    #20s for creating the amplitude values for the desired values of the system
    for b in range(4) :
    #5s
        for a in range(50) :
            desired_x.append(0)
        for a in range(50) :
            desired_x.append(1)
        for a in range(50) :
            desired_x.append(2)
        for a in range(50) :
            desired_x.append(3)
        for a in range(50) :
            desired_x.append(2)
    for a in range(1) :
            desired_x.append(2)
    #"""
    """
    # Get x values of the sine wave
    timearray        = np.arange(0, 20.02, 0.02);
    # Amplitude of the sine wave is sine of a variable like time
    desired_x   = np.sin(timearray)
    #"""
    
    dt = 0.02 #s - 50Hz
    time_steps = []
    time_curr = -0.02 #Has to change with dt
    m = 2 #kg
    c = 10 #Ns/m - damping coefficient
    k = 20 #N/s - Spring coefficient 
    MSDnum = [0, 0, 1] #M-S-D system definition
    MSDden = [m, c, k]
    pop = []
    fit_val = []
    return desired_x, dt, time_steps, time_curr, MSDden, MSDnum, pop, fit_val


###### MAIN PROGRAM CALL BODY ######
desired_x, dt, time_steps, time_curr, MSDden, MSDnum, pop, fit_val = initialise()
log_time = 0.0

logged_time = []
changes = 0
y_global = []
a = -0.02 #has to change with dt

logged_time.append(time_steps.copy())
while time_curr < 20 :

    a = round(a + dt, 2)
    time_steps.append(a) 
    time_curr = round(time_curr + dt, 2)
    
    """ DISTURBANCES
    if time_curr == 5.8 :
        m = 2
        c = 28 # Damper gets damaged
        k = 20
        MSDden = [m, c, k]
    #"""
    """
    if time_curr == 14.8 :
        m = 2
        c = 40 #Damper damage increases
        k = 5 #Spring gets damaged
        MSDden = [m, c, k]
    #"""

    if time_curr == 0 :
        pop = GA_controller(time_steps, pop, fit_val) #It is currently expected that on start-up performance will be awful
        #run through output sim
    if len(time_steps) > 3 : #Set at 3 because of current recognise detection length. Can be adjusted if necessary
        recognise, changes, y_global, logged_time = Recognise(changes, y_global, time_curr, log_time, time_steps, MSDnum, MSDden, pop, desired_x)
        #if recognise == True
        #fitness test all pops from time @ logged[changed] to time_steps, sort, boot up the GA controller (or other method) for better genome
        #fitness test all pops from time @ logged[changed] to time_steps, sort and top is then chosen for use in recognise
        if recognise == True :
            log_time = time_curr
            fit_val = Fitness(MSDnum, MSDden, pop, time_steps, desired_x, y_global, logged_time)
            pop, fit_val = Fit_sort(pop, fit_val)
            pop = GA_controller(time_steps, pop, fit_val)
            fit_val = Fitness(MSDnum, MSDden, pop, time_steps, desired_x, y_global, logged_time)
            pop, fit_val = Fit_sort(pop, fit_val)
            

print("Number of controller changes: ", changes)
print("Top genome: kd ", pop[0][0], "  kp ", pop[0][1], "  ki ", pop[0][2])

plt.plot(time_steps, desired_x, label = "Desired Output - X_d(s)")
plt.plot(time_steps, y_global, label = "System Output - X(s)") 
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
