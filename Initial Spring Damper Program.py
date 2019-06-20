''' The maths in this has not been verified yet '''
''' There was no division of Mass of the damper in the mass spring damper equation.  '''

import pylab as pylab


from scipy.integrate import odeint
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots

from pylab import cos, pi, arange, sqrt, pi, array, array

def MassSpringDamper(state,t):
    '''
    k=spring constant, Newtons per metre
    m=mass, Kilograms
    c=dampign coefficient, Newton*second / meter    
    
    for a mass,spring
        xdd = ((-k*x)/m) + g
    (for a mass, spring, damper 
        xdd = -k*x/m -c*xd-g)
        correction - xdd = - (c*xd +k*x)/m - g
    for a mass, spring, damper with forcing function
        xdd = -k*x/m -c*xd-g + cos(4*t-pi/4)
    '''
  
    k=124e3  # spring constant, kN/m
    m=64.2 # mass, Kg
    c=3  # damping coefficient 
    # unpack the state vector
    x,xd = state # displacement,x and velocity x'
    g = 9.8 # metres per second**2
    # compute acceleration xdd = x''
    # omega = 1.0 # frequency
    # phi = 0.0 # phase shift
    # A = 5.0 # amplitude
    # old function xdd = -k*x/m -c*xd-g + A*cos(2*pi*omega*t - phi)
    xdd = - (c*xd + k*x/m) 
    return [xd, xdd]


state0 = [0.0, 1.2]  #initial conditions [x0 , v0]  [m, m/sec] 
ti = 0.0  # initial time
tf = 1.0  # final time
step = 0.001  # step
t = arange(ti, tf, step)
state = odeint(MassSpringDamper, state0, t)
x = array(state[:,[0]])
xd = array(state[:,[1]])


# Plotting displacement and velocity
pylab.rcParams['figure.figsize'] = (15, 12)
pylab.rcParams['font.size'] = 18

fig, ax1 = pylab.subplots()
ax2 = ax1.twinx()
ax1.plot(t,x*1e3,'b',label = r'$x (mm)$', linewidth=2.0)
ax2.plot(t,xd,'g--',label = r'$\dot{x} (m/sec)$', linewidth=2.0)
ax2.legend(loc='lower right')
ax1.legend()
ax1.set_xlabel('time , sec')
ax1.set_ylabel('disp (mm)',color='b')
ax2.set_ylabel('velocity (m/s)',color='g')
pylab.title('Mass-Spring System with $V_0=1.2$ and $\delta_{max}=22.9mm$')
pylab.show()