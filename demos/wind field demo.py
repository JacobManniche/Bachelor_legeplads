import matplotlib.pyplot as plt
import numpy as np
from Tracer.windfield import WindField


# example configuration
wf = WindField(nx=500, ny=100, nz=100, profile="log",U_ref=10,z0=0.03)

point =wf.get_point(20,10,10)
print(point)

#--------------------------------------------------------------------
#Plotting

#fig = plt.figure(figsize=(12,8))
#ax = fig.add_subplot(111, projection='3d')
# Downsample the field for 3D plotting
#step = 10
#x = np.arange(0, wf.nx, step)
#y = np.arange(0, wf.ny, step)
#z = np.arange(0, wf.nz, step)
#X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

#U = wf.velocity[::step, ::step, ::step, 0]
#V = wf.velocity[::step, ::step, ::step, 1]
#W = wf.velocity[::step, ::step, ::step, 2]

#ax.quiver(X, Y, Z, U, V, W, length=20, normalize=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()
#--------------------------------------------------------------------
