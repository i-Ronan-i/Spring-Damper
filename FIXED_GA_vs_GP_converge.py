""" Genetic Algorithm Tuning of a PID controller 
for offline optimisation of a Mass-Spring-Damper System. """

import numpy as np
from scipy.integrate import solve_ivp, RK23, odeint
from scipy.interpolate import PchipInterpolator
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time

def create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    global flag
    """Creates the initial population of the genetic algorithm while making sure it adheres to force constraints"""
    for s in range(pop_num):
        flag = True
        flagged = 0
        while flag == True:
            flag = False
            #Creating the random PID values
            kd_cur = round(random.uniform(kd_min, kd_max), 2)
            kp_cur = round(random.uniform(kp_min, kp_max), 2)
            ki_cur = round(random.uniform(ki_min, ki_max), 2) 
            Kp = kp_cur
            Ki = ki_cur
            Kd = kd_cur
            
            #Simulates the current system [s].
            def sysCrInitPID(t,x):
                global force_constraint
                global flag

                r=set_interp(t)

                #State Variables
                y = x[0]            # x1 POSITION
                dydt = x[1]         # x2 VELOCITY
                yi = x[2]           # x3

                u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output
                if abs(u) > force_constraint:
                    print("u, t: ", u, t)
                    print(Kp, Ki, Kd)
                    flag = True


                dxdt = [0,0,0]

                dxdt[0] = dydt
                dxdt[1] = (- c * dydt - k * y + u)/m
                dxdt[2] = r - y

                return [dxdt[0],dxdt[1],dxdt[2]]

            #tev = np.linspace(0,20,1000)
            tev = []
            temp = round(0.0, 2)
            for times in range(int(20/dt)):
                tev.append(temp)
                temp = round(temp + dt, 2)
            x_ini = [0,0,0]        # initial conditions
            solga = solve_ivp(sysCrInitPID, [0, 20], x_ini, t_eval=tev)
            #If system never exceeds max force then continue.
        
        #Into 2-D List. Access via pop[i][j]
        if flag == False:
            pop.insert(s, [kd_cur, kp_cur, ki_cur])
    return pop

def fitness(pop):
    """Calculates the fitness values of each member in pop[population]
    Also sets the simulation time in timesteps."""
    fit_val = []
    for s in range(len(pop)):
        #Grab the pop PID values
        Kp = pop[s][1]
        Ki = pop[s][2]
        Kd = pop[s][0]
        
        #Simulates the current system [s].
        def sysFitPID(t,x):
            global force_constraint

            r=set_interp(t)

            #State Variables
            y = x[0]            # x1 POSITION
            dydt = x[1]         # x2 VELOCITY
            yi = x[2]           # x3

            u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output
            if abs(u) > force_constraint:
                print("Fitness Force Constraint ALERT pop", s)

            dxdt = [0,0,0]

            dxdt[0] = dydt
            dxdt[1] = (- c * dydt - k * y + u)/m
            dxdt[2] = r - y

            return [dxdt[0],dxdt[1],dxdt[2]]

        #tev = np.linspace(0,20,1000)
        tev = []
        temp = round(0.0, 2)
        for times in range(int(20/dt)):
            tev.append(temp)
            temp = round(temp + dt, 2)
        x_ini = [0,0,0]        # initial conditions
        solga = solve_ivp(sysFitPID, [0, 20], x_ini, t_eval=tev)
        y_out = solga.y[0, :]
        t_out = solga.t

        err_val = 0.0
        for y in range(len(t_out)) :
            err_val = err_val + abs(set_interp(t_out[y]) - y_out[y])  

        fit_val.insert(s, err_val)
    return fit_val
    
def crossover(a, b):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    global flag
    new_a = []  #Clearing previous 
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut

    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]

    #Creates the new crossed-over list
    new_a = new_a1 + new_a2
    #trying to keep force constraints upheld
    
    Kp = new_a[1]
    Ki = new_a[2]
    Kd = new_a[0]
    #Simulates the current system [s].
    flag = False
    def sysCrossPID(t,x):
        global force_constraint
        global flag

        r=set_interp(t)

        #State Variables
        y = x[0]            # x1 POSITION
        dydt = x[1]         # x2 VELOCITY
        yi = x[2]           # x3

        u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output
        if abs(u) > force_constraint:
            flag = True

        dxdt = [0,0,0]

        dxdt[0] = dydt
        dxdt[1] = (- c * dydt - k * y + u)/m
        dxdt[2] = r - y

        return [dxdt[0],dxdt[1],dxdt[2]]

    #tev = np.linspace(0,20,1000)
    tev = []
    temp = round(0.0, 2)
    for times in range(int(20/dt)):
        tev.append(temp)
        temp = round(temp + dt, 2)
    x_ini = [0,0,0]        # initial conditions
    solga = solve_ivp(sysCrossPID, [0, 20], x_ini, t_eval=tev)    

    #If system exceeds max force then crossover is aborted.
    if flag == True:
        new_a = a
    return new_a

def mutate(pop, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max) :
    """Takes current population member and add a probability chance
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%."""
    global flag
    global kpbig, kibig, kdbig
    pop_curr = pop
    for i in range(0, len(pop_curr)):
        mutation = False 
        for o in range(3) :
            if random.random() <= mut_prob:
                mutation = True
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
                        
        #Keeping force contraints upheld
        if mutation == True:
            flag = True
            while flag == True:
                flag = False
                #Creating the random PID values
                Kp = pop_curr[i][1]
                Ki = pop_curr[i][2]
                Kd = pop_curr[i][0]
                kpbig = 0
                kibig = 0
                kdbig = 0
                #Simulates the current system [s].
                def sysMutPID(t,x):
                    global force_constraint
                    global flag
                    global kpbig, kibig, kdbig

                    r=set_interp(t)

                    #State Variables
                    y = x[0]            # x1 POSITION
                    dydt = x[1]         # x2 VELOCITY
                    yi = x[2]           # x3

                    u = Kp * (r - y) + Ki * yi - Kd * dydt         # PID output
                    if abs(u) > force_constraint:
                        flag = True

                    #counters for mutation reduction if force is exceeded
                    kp_force = Kp*(r-y)
                    ki_force = Ki*yi
                    kd_force = -Kd*dydt
                    if kp_force > ki_force and kp_force > kd_force :
                        kpbig +=1
                    elif ki_force > kp_force and ki_force > kd_force :
                        kibig +=1
                    elif kd_force > ki_force and kd_force > kp_force :
                        kdbig +=1

                    dxdt = [0,0,0]

                    dxdt[0] = dydt
                    dxdt[1] = (- c * dydt - k * y + u)/m
                    dxdt[2] = r - y
                    return [dxdt[0],dxdt[1],dxdt[2]]

                #tev = np.linspace(0,20,1000)
                tev = []
                temp = round(0.0, 2)
                for times in range(int(20/dt)):
                    tev.append(temp)
                    temp = round(temp + dt, 2)
                x_ini = [0,0,0]        # initial conditions
                solga = solve_ivp(sysMutPID, [0, 20], x_ini, t_eval=tev)

                #If system never exceeds max force then continue.
                if flag == True:
                    #Mutation reduction of biggest contributing factor
                    if kpbig >= kibig and kpbig >= kdbig:
                        pop_curr[i][1] = round(pop_curr[i][1] * 0.99,2)
                    elif kibig > kpbig and kibig >= kdbig:
                        pop_curr[i][2] = round(pop_curr[i][2] * 0.99,2)
                    elif kdbig > kibig and kdbig > kpbig:
                        pop_curr[i][0] = round(pop_curr[i][0] * 0.99,2)
    return pop_curr

def create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #Saves top 3 performing genomes
    global flag
    pop_top = []
    for n in range(1) :
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
    for s in range(pop_num - len(pop_new)):
        flag = True
        while flag == True:
            flag = False
            #Creating the random PID values
            kd_cur = round(random.uniform(kd_min, kd_max), 2)
            kp_cur = round(random.uniform(kp_min, kp_max), 2)
            ki_cur = round(random.uniform(ki_min, ki_max), 2) 
            Kp = kp_cur
            Ki = ki_cur
            Kd = kd_cur
            
            #Simulates the current system [s].
            def sysCrGenPID(t,x):
                global force_constraint
                global flag

                r=set_interp(t)

                #State Variables
                y = x[0]            # x1 POSITION
                dydt = x[1]         # x2 VELOCITY
                yi = x[2]           # x3

                u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output
                if abs(u) > force_constraint:
                    flag = True

                dxdt = [0,0,0]

                dxdt[0] = dydt
                dxdt[1] = (- c * dydt - k * y + u)/m
                dxdt[2] = r - y

                return [dxdt[0],dxdt[1],dxdt[2]]

            #tev = np.linspace(0,20,1000)
            tev = []
            temp = round(0.0, 2)
            for times in range(int(20/dt)):
                tev.append(temp)
                temp = round(temp + dt, 2)
            x_ini = [0,0,0]        # initial conditions
            solga = solve_ivp(sysCrGenPID, [0, 20], x_ini, t_eval=tev)
            #If system never exceeds max force then continue.
        
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


"""""""""""""""""""""""""""""""""---------- MAIN CODE PROGRAM BODY ----------------"""""""""""""""""""""""""""""""""""""""
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

start_time = time.time()
#first thing is to create the system itself - MSD, input
#MSD state definitions
m = 1 # kg   - mass
c = 10 # Ns/m - damping coefficient
k = 20 # N/m  - spring coefficient
pop_num = 60    #How large the initial population is
pop = []
mut_prob = 0.08  #probability for mutation set here
dt = 0.02
iteration_max = 60 #Total number of iterations and generations set here
flag = True #Making it global
kpbig = 0   #Globalising these 
kibig = 0   #
kdbig = 0   #

#Minimum and maximum PID coefficient gains.
kd_min, kd_max = 0, 500
kp_min, kp_max = 0, 500
ki_min, ki_max = 0, 900

def setpoint(t):
    if (t >= 0 and t < 4):
            r = 2
    elif (t >= 4 and t < 9):
            r = 3
    elif (t >= 9 and t < 14):
            r = 8
    elif (t >= 14 and t < 21):
            r = 3
    else:
            r = 0
    return r


tempo = np.linspace(0, 20, 20*5)
rise_time = 0.1
set_point=[]
for items in tempo:
    set_point.append(setpoint(items))

set_interp = PchipInterpolator(tempo, set_point)

rise_time = 0.1
r_max = max(set_point)
r_min = min(set_point)
ar = 2 * r_max / (rise_time ** 2)
vr = ar * rise_time
yr = 0.9 * r_max

force_constraint = m * ar + c * vr + k * yr
print("Force constraint is (N): ", force_constraint)

################################################# ----- Main GA call stack ----- ################################################
iteration = 0
while iteration < iteration_max:
    print("Iteration: ", iteration)
    if iteration == 0:
        pop = create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
        fit_val = fitness(pop)
        iteration = iteration + 1

    if iteration < iteration_max and iteration > 0:
        pop, fit_val = fit_sort(pop, fit_val)
        pop = create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
        fit_val = fitness(pop)
        iteration = iteration + 1



"""This is the final section with the top solution being chosen and used"""
#Final simulation run
fit_val = fitness(pop)
pop, fit_val = fit_sort(pop, fit_val)
print("Top overall Coefficients: kd = ", pop[0][0], "  kp = ", pop[0][1], "  ki = ", pop[0][2])
print("Fitness value of top performing member: ", round(fit_val[0], 4))
print("Time elapsed: ", time.time()-start_time)
Kp = pop[0][1]
Ki = pop[0][2]
Kd = pop[0][0]

#Simulates the current system [s].
def sysFinPID(t,x):
    global force_constraint

    r=set_interp(t)
    #State Variables
    y = x[0]            # x1 POSITION
    dydt = x[1]         # x2 VELOCITY
    yi = x[2]           # x3

    u = Kp * (r - y) + Ki * yi - Kd * dydt         #PID output

    if abs(u)>force_constraint:
        print("FINAL SOLUTION A L E R T")

    dxdt = [0,0,0]

    dxdt[0] = dydt
    dxdt[1] = (- c * dydt - k * y + u)/m
    dxdt[2] = r - y

    return [dxdt[0],dxdt[1],dxdt[2]]

#tev = np.linspace(0,20,1000)
temp = round(0.00, 2)
tev = []
for times in range(int(20/dt)):
    tev.append(temp)
    temp = round(temp + dt, 2)
x_ini = [0,0,0]        # initial conditions
solga = solve_ivp(sysFinPID, [0, 20], x_ini, t_eval=tev)
y_out = solga.y[0, :]
t_out = solga.t

err_vall = 0.0
for y in range(len(t_out)) :
    err_vall = err_vall + abs(set_interp(t_out[y]) - y_out[y])
err_val = err_vall 
print("The Fitness Value of the solution is: ", err_val) 

#Plotting code
plt.plot(tempo, set_point, "r--", label="Set Point Command [m]")
plt.plot(t_out, y_out, label = "System Output X(s) [m]") 
plt.xlabel('Time [s]')
# Set the y axis label of the current axis.
plt.ylabel('Amplitude [m]')
# Set a title of the current axes.
#plt.title('System Response to Varying Step Inputs')
plt.title('System Response over 20 Seconds.')

plt.xticks(np.arange(0, 20.5, step=1))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()