#This code calculates a 3D trajectory of an arbitrarily spinning ball
#subject to gravity, aerodynamic drag, and the Magnus force
import numpy as np
import math
import pandas as pd
#drag coefficient from v and S according to Clanet 2015
def cdfunc(vx,vy,vz,w,r,vc,vs):
    v=math.sqrt(vx**2+vy**2+vz**2)
    S=w*r/v
    if S>0.05 and v>vc:
        cd=0.4127*S**0.3056
    else:
        cd=0.155+0.346/(1+math.exp((v-vc)/vs))
    return cd
#Function for the numeric integration
def numericintegration(vx0,vy0,vz0,x0,y0,z0,wx,wy,wz,vc,vs,r,m,cl):
    kt=0
    #Total angular frequency
    w=math.sqrt(wx**2+wy**2+wz**2)
    #time step size (vary to check the convergence)
    dto1=1/220
    #Initiating position, time and velocity
    xo1=np.array([x0])
    yo1=np.array([y0])
    zo1=np.array([z0])
    vxo1=np.array([vx0])
    vyo1=np.array([vy0])
    vzo1=np.array([vz0])
    to1=np.array([0])
    #step counter
    ko1=1
    #First order numeric integration
    while min(yo1)>-0.001:  #a little less then zero to take out the initial point with z=0.
        cd=cdfunc(vxo1[-1],vyo1[-1],vzo1[-1],w,r,vc,vs)
        vo1=math.sqrt(vxo1[-1]**2+vyo1[-1]**2+vzo1[-1]**2)
        axo1=-1/(2*m)*cd*A*rho*vo1**2*vxo1[-1]/vo1\
        +cl/m*math.pi*r**3*rho*(wy*vzo1[-1]-wz*vyo1[-1])
        ayo1=-1/(2*m)*cd*A*rho*vo1**2*vyo1[-1]/vo1-g\
        +cl/m*math.pi*r**3*rho*(wz*vxo1[-1]-wx*vzo1[-1])
        azo1=-1/(2*m)*cd*A*rho*vo1**2*vzo1[-1]/vo1\
            +cl/m*math.pi*r**3*rho*(wx*vyo1[-1]-wy*vxo1[-1])
        vxo1=np.append(vxo1,np.array([vxo1[-1]+axo1*dto1]))
        vyo1=np.append(vyo1,np.array([vyo1[-1]+ayo1*dto1]))
        vzo1=np.append(vzo1,np.array([vzo1[-1]+azo1*dto1]))
        xo1=np.append(xo1,np.array([xo1[-1]+vxo1[-1]*dto1]))
        yo1=np.append(yo1,np.array([yo1[-1]+vyo1[-1]*dto1]))
        zo1=np.append(zo1,np.array([zo1[-1]+vzo1[-1]*dto1]))
        to1=np.append(to1,np.array([to1[-1]+dto1]))
        
        ko1=ko1+1


    data=np.array([to1,xo1,yo1,zo1,vxo1,vyo1,vzo1])
    
    return data
#Constants
g=9.81
rho=1.2 

#ball
r=0.11    #radius
A=math.pi*r**2 #projected or frontal area of the ball [m^2]
m=0.430         #kg


#default drag and lift coefficients 
cd=0.15         #default ball drag coefficient
cl=1.7/math.pi       #lift coefficient 


#For the CD-model
vc=12.19 
vs=1.309
#Starting position
x0=0
y0=0
z0=0
#Starting angular velocity
wx=0
wy=2*math.pi*13.5
wz=0
w=math.sqrt(wx**2+wy**2+wz**2)
#Roberto Carlos 1997 :13.5 rps, 38 m/s
#initial velocity
v0=40                    #initial velocity in the (x,y) plane
theta0=45                #initial angle to the horizontal
vz0=0                    #initial velocity component in z direction
vtot0=math.sqrt(v0**2+vz0**2)   #initial total speed of the ball
vx0=v0*math.cos(theta0*math.pi/180) #In cartesian coordinates
vy0=v0*math.sin(theta0*math.pi/180)
#non-dimensional parameters
Ap=v0**2/(r*g)                    #range parameter A
B=m*g/(0.5*cd*A*rho*v0**2)   #ballistic parameter B
S=w*r/vtot0                     #spin parameter S


Thang=2*(vy0/g)
tvec=np.linspace(0,Thang,2200)
#Run the function with the given data (You can make several initial conditions...
#and compare them. Just make sure to change the input and give the output another name eg.)
#data2=numericintegration(vx02,vy02,vz02,x02,y02,z02,wx2,wy2,wz2,vc,vs,r,m,cl)
#Remember to define the new variables first.
data=numericintegration(vx0,vy0,vz0,x0,y0,z0,wx,wy,wz,vc,vs,r,m,cl)
#Extract the x,y,z and t-values
t = data[0,:]
x = data[1,:]
y = data[2,:]
z = data[3,:]
rangexo1 = max(data[1,:])
rangezo1 = max(data[2,:])
heighto1 = max(data[3,:])
hangtimeo1=max(data[0,:])
import matplotlib.pyplot as plt # Import the matplotlib.pyplot module
#Compare vacuum and numeric air solution
plt.figure(10)
plt.subplot(2,1,1)
plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major')
plt.axis('equal')
plt.box
plt.plot(x, y, "b-",linewidth=4)
#Analytic
plt.plot(vx0*tvec, vy0*tvec-1/2*g*(tvec)**2, "g-",linewidth=2)
#Measured is plotted twice to make sure the data can be viewed
plt.xlabel("x [m]") # Set the x-axis label
plt.ylabel("y [m]") # Set the y-axis label
plt.legend(['Num,air','Ana,vac'],loc='best',ncol=3)
plt.subplot(2,1,2)
plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major')
plt.axis('equal')
plt.box
plt.plot(x, z, "b-",linewidth=4)
#Analytic
plt.plot(vx0*tvec, vz0*tvec, "g-",linewidth=2)
#Measured is plotted twice to make sure the data can be viewed
plt.xlabel("x [m]") # Set the x-axis label
plt.ylabel("z [m]") # Set the y-axis label
plt.legend(['Num,air','Ana,vac'],loc='best',ncol=3)
plt.show()
