#importing stuff
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
from pylab import cos, pi, arange, sqrt, pi, array, array

#setting time steps
steps2 = 0.001
t0 = 0
tf = 1
t = arange(t0, tf, steps2)

#for setting the displacement values
des_x = list()
for i in range(0,200):
    des_x.append(10.0)
for i in range(200,400):
    des_x.append(0.0)
for i in range(400,600):
    des_x.append(10.0)
for i in range(600,800) :
    des_x.append(0.0)
for i in range(800,1000):
    des_x.append(10.0)

#plotting stolen from main program just for visualisation.
pylab.rcParams['figure.figsize'] = (10, 8)
pylab.rcParams['font.size'] = 12

fig, ax1 = pylab.subplots()
ax1.plot(t, des_x,'b', label = r'$x (mm)$', linewidth=2.0)
ax1.legend()
ax1.set_xlabel('time (seconds)')
ax1.set_ylabel('disp (mm)',color='b')
pylab.title('Desired Displacement Plot')
pylab.show()