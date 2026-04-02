#3D bounce motion of an arbitrarily spinning ball
import numpy as np
import math

# gravity
g=9.81      #[m/s^2]

#ball
m=0.43       #mass [kg]
r=0.1        #ball radius [m]
a=2/3        #coefficient of I: 2/3 for hollow sphere, 2/5 for solid sphere
I=a*m*r**2    #moment of inertia

#horizontal coefficients of restitution in x and z
#and verical coefficient of restitution 
#ey is between 0 and 1
#ex is between -1 and 1. -1: no friction, 0: rolling, completely inelastic,
#1 completely elastic collision that reverse sign of the velocity of the
#contact point, between -1 and 0 is between gliding and rolling
ex=0.8  
ey=.95     
ez=ex   

k=0 #bounce number-1

#start position
xb=np.array([0])   
yb=np.array([0])
zb=np.array([0])

#velocity before the first bounce (before: b)
vxb=np.array([1])
vyb=np.array([-1])
vzb=np.array([0])

#spin vector before first bounce
wxb=np.array([0])
wyb=np.array([0])
wzb=np.array([10])

#vacuum solution of the parabola
hangtime0=2*vyb[k]/g
height0=vyb[k]**2/(2*g)
rangex0=vxb[k]*hangtime0
rangez0=vzb[k]*hangtime0

tvac0 = np.linspace(0,hangtime0)
xvac0 = vxb[k]*tvac0
yvac0 = vyb[k]*tvac0-g/2*tvac0**2
zvac0 = vzb[k]*tvac0

#loop over bounces
kvec=range(0,19)
vvecb=np.zeros((6,np.size(kvec)+1))
vvecb[:,0]=np.array([vxb[k], vyb[k], vzb[k],wxb[k], wyb[k], wzb[k]])
#velocity and spin after the kth bounce    
# vxa(k)=(1-a*ex)/(a+1)*vxb(k)-a*(ex+1)/(a+1)*r*wzb(k)
# vya(k)=-ey*vyb(k)
# vza(k)=(1-a*ez)/(a+1)*vzb(k)+a*(ez+1)/(a+1)*r*wxb(k)
# 
# wxa(k)=(a-ez)/(a+1)*wxb(k)+(ez+1)/(a+1)/r*vzb(k)
# wya(k)=wyb(k)
# wza(k)=(a-ex)/(a+1)*wzb(k)-(ex+1)/(a+1)/r*vxb(k)

#bounce matrix
BOUNCE=np.array([[(1-a*ex)/(a+1),0,0,0,0,-a*(ex+1)/(a+1)*r],\
    [0,-ey,0,0,0,0],\
    [0,0,(1-a*ez)/(a+1),a*(ez+1)/(a+1)*r,0,0],\
    [0,0,(ez+1)/(a+1)/r,(a-ez)/(a+1),0,0],\
    [0,0,0,0,1,0],\
    [-(ex+1)/(a+1)/r,0,0,0,0,(a-ex)/(a+1)]])

#air matrix
AIR=np.eye(6)
AIR[1,1]=-1
#initiating vectors and matrixes
vveca=np.zeros((6,np.size(kvec)))
hangtime=np.zeros((np.size(kvec),1))
tvac=np.zeros((np.size(np.linspace(0,hangtime[0])),np.size(kvec)))
xvac=np.zeros((np.size(np.linspace(0,hangtime[0])),np.size(kvec)))
yvac=np.zeros((np.size(np.linspace(0,hangtime[0])),np.size(kvec)))
zvac=np.zeros((np.size(np.linspace(0,hangtime[0])),np.size(kvec)))
rangez=np.zeros((np.size(kvec),1))
rangex=np.zeros((np.size(kvec),1))

for k in (kvec):
    #calculate conditions after bounce from before bounce
    
    vveca[:,k]=np.dot(BOUNCE,vvecb[:,k])

    #get velocities for easier coding
    vx=vveca[0,k]
    vy=vveca[1,k]
    vz=vveca[2,k]

    #calculate range in x and z and hangtime after the kth bounce
    
    hangtime[k]=2*vy/g
    rangex[k]=vx*hangtime[k]
    rangez[k]=vz*hangtime[k]

    #calculate vacuum trajectoriy
    tvac[:,k] = np.linspace(0,hangtime[k]).flatten()
    xvac[:,k] = xb[k]+vx*tvac[:,k]
    yvac[:,k] = vy*tvac[:,k]-g/2*tvac[:,k]**2
    zvac[:,k] = zb[k]+vz*tvac[:,k]
    
    #calculate conditions before (k+1)th bounce
    #in vacuum only the sign of vy reverses
    vvecb[:,k+1]=(np.dot(AIR,vveca[:,k]))
    
    #calculate new bounce point
    xb=np.append(xb,[xb[k]+rangex[k]])
    zb=np.append(zb,[zb[k]+rangez[k]])

#calculate the energy, the force integrals and verify the horizontal
#coeefficients of restitution
vxa=vveca[0,:]vya=vveca[1,:]vza=vveca[2,:]
vxb=vvecb[0,:]vyb=vvecb[1,:]vzb=vvecb[2,:]
wxa=vveca[3,:]wya=vveca[4,:]wza=vveca[5,:]
wxb=vvecb[3,:]wyb=vvecb[4,:]wzb=vvecb[5,:]


E=0.5*m*(vxa**2+vya**2+vza**2+r**2*(wxa**2+wya**2+wza**2))
FxdtF=m*(vxa-vxb[0:-1])
FydtF=m*(vya-vyb[0:-1])
FzdtF=m*(vza-vzb[0:-1])

FxdtM=I*(wza-wzb[0:-1])/r
FzdtM=-I*(wxa-wxb[0:-1])/r

checkex=-(vxa+r*wza)/(vxb[0:-1]+r*wzb[0:-1])
checkez=-(vza-r*wxa)/(vzb[0:-1]-r*wxb[0:-1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#figures
import matplotlib.pyplot as plt # Import the matplotlib.pyplot module
plt.figure(10)
plt.subplot(2,1,1)
plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major')
plt.plot(xvac0,yvac0)
plt.plot(xvac,yvac)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.title('x-y')

plt.subplot(2,1,2)
plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major')
plt.plot(zvac0,yvac0)
plt.plot(zvac,yvac)
plt.xlabel('z [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.title('z-y')

from mpl_toolkits import mplot3d
plt.figure(20)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#plt.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in', which='major')
#zdir('y') makes the y-direction point up!
ax.plot3D(xvac0,yvac0,zvac0,zdir='y')
#plt.show()
for k in kvec:
    ax.plot3D(xvac[:,k],yvac[:,k],zvac[:,k],zdir='y')
plt.show()


plt.figure(30)
plt.subplot(2,3,1)
plt.plot(kvec,vxa,'o')
plt.xlabel('bounce [-]')
plt.ylabel('vx [m/s]')
plt.subplot(2,3,2)
plt.plot(kvec,vza,'o')
plt.xlabel('bounce [-]')
plt.ylabel('vz [m/s]')
plt.subplot(2,3,3)
plt.plot(kvec,wxa,'o')
plt.xlabel('bounce [-]')
plt.ylabel('wx [1/s]')
plt.subplot(2,3,4)
plt.plot(kvec,wza,'o')
plt.xlabel('bounce [-]')
plt.ylabel('wz [1/s]')
plt.subplot(2,3,5)
plt.plot(kvec,E,'o')
plt.xlabel('bounce [-]')
plt.ylabel('E/m [J/kg]')

plt.figure(40)
plt.subplot(1,3,1)
plt.plot(kvec,FxdtF,'o')
plt.plot(kvec,FxdtM,'x')
plt.xlabel('bounce [-]')
plt.ylabel('Fxdt [Ns]')
plt.subplot(1,3,2)
plt.plot(kvec,FydtF,'o')
plt.xlabel('bounce [-]')
plt.ylabel('Fydt [Ns]')
plt.subplot(1,3,3)
plt.plot(kvec,FzdtF,'o')
plt.plot(kvec,FzdtM,'x')
plt.xlabel('bounce [-]')
plt.ylabel('Fzdt [Ns]')
