import numpy as np
import matplotlib.pyplot as plt
from tracer import initial_velocity, solver


def plot_trajectory(P0, V0, W0, wind):

    data = solver(P0, V0, W0, wind)

    #3d plot of the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    plot_range = np.abs(data[1]).max()*1.1

    x_plane = np.linspace(-plot_range, plot_range, 10)
    y_plane = np.linspace(-plot_range, plot_range, 10)

    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    Z_p = np.zeros_like(X_p) # The plane is at height Z=0
    ax.plot_surface(X_p, Y_p, Z_p, color='tab:green', alpha=1, antialiased=False, zorder=0) 


    ax.plot(P0[0], P0[1], P0[2], 'ro', zorder=1, label='Initial Position') # plot the initial position
    ax.plot(data[1][-1,0], data[1][-1,1], data[1][-1,2], 'kx', zorder=1, label='Final Position') # plot the final position
    ax.plot(data[1][:,0], data[1][:,1], data[1][:,2], zorder=1, color='tab:orange', label='Trajectory') # plot the trajectory
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_zlim(0, data[1][:,2].max()*1.1)
    ax.set_aspect('equal', 'box') # set equal aspect ratio for x and y axes
    ax.set_title('3D Trajectory of the Ball')

    plt.show()


if __name__ == "__main__":
    # This is where we will run our simulation and plot the results
    # We can change the initial conditions and parameters here to see how they affect the trajectory of the ball
    P0 = [0,0,0]
    V0 = initial_velocity(speed=50, angle=65) 
    W0 = np.array([0, 0, 100])
    wind = np.array([-2, -2, 10]) # 

    plot_trajectory(P0, V0, W0, wind)
