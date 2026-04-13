from Tracer import WindField, solver, initial_velocity

from Tracer.tracer import norm, coefficients, euler_solver
import numpy as np
import matplotlib.pyplot as plt

stats= [{'height':32, 'land_angle':39, 'distance':258}]
stats.append({'height':29, 'land_angle':52, 'distance':130})


def plot_trajectory(trajectories):
    ax = plt.subplots(3, 1, figsize=(10, 8))[1]
    for t, p, v, w in trajectories:
        ax[0].plot(p[:, 0], p[:, 2]) # Plot height vs time
        ax[1].plot(p[:, 0], p[:, 1]) # Plot horizontal distance vs time
        ax[2].plot(t, w[:, 1], label='Spin Rate (rad/s)')
    
    # ax[0].axhline(stats[0]['height'], color='tab:blue', linestyle='--', label='PGA Average Height')
    # ax[0].axhline(stats[1]['height'], color='tab:orange', linestyle='--', label='PGA Average Height')
    # ax[0].axvline(stats[0]['distance'], color='tab:blue', linestyle='--', label='PGA Average Distance')
    # ax[0].axvline(stats[1]['distance'], color='tab:orange', linestyle='--', label='PGA Average Distance')

    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Height (m)')
    ax[0].set_title('Trajectory of the Golf Ball')
    ax[0].grid()
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Horizontal Distance (m)')
    ax[1].set_title('Horizontal Trajectory of the Golf Ball')
    ax[1].grid()
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Spin Rate (rad/s)')
    ax[2].set_title('Spin Rate Decay Over Time')
    ax[2].grid()
    ax[2].legend()
    plt.tight_layout()
    plt.show()

def plot_coefficients(trajectories):
    mu = 1.82e-5 # Dynamic viscosity of air (kg/(m·s))
    r = 0.0214 # Radius of the golf ball (m)
    
    ax = plt.subplots(2, 2, figsize=(10, 8))[1]
    for t, p, v, w in trajectories:
        re = [(norm(v[i]) * (2*r)) / mu for i in range(len(v))] # Reynolds Number calculation
        s = [norm(w[i]) * r / norm(v[i]) if norm(v[i]) > 0 else 0 for i in range(len(v))] # Spin parameter
        cd = [coefficients(norm(v[i]), norm(w[i]), eske=False)[0] for i in range(len(re))]
        cl = [coefficients(norm(v[i]), norm(w[i]), eske=False)[1] for i in range(len(re))]
        
        ax[0, 0].plot(re, cd) # Plot height vs time
        ax[1, 0].plot(s, cl) # Plot horizontal distance vs time

        ax[0, 1].plot(t, cd) # Plot height vs time
        ax[1, 1].plot(t, cl) # Plot horizontal distance vs time
    
    ax[0, 0].set_xlabel('Reynolds Number')
    ax[0, 0].set_ylabel('Drag Coefficient (Cd)')
    ax[0, 0].set_title('Drag Coefficient vs Reynolds Number')
    ax[0, 0].grid()
    ax[1, 0].set_xlabel('Spin Parameter (s)')
    ax[1, 0].set_ylabel('Lift Coefficient (Cl)')
    ax[1, 0].set_title('Lift Coefficient vs Spin Parameter')
    ax[1, 0].grid()
    ax[0, 1].set_xlabel('Time (s)')
    ax[0, 1].set_ylabel('Drag Coefficient (Cd)')
    ax[0, 1].set_title('Drag Coefficient Decay Over Time')
    ax[0, 1].grid()
    ax[1, 1].set_xlabel('Time (s)')
    ax[1, 1].set_ylabel('Lift Coefficient (Cl)')
    ax[1, 1].set_title('Lift Coefficient Decay Over Time')
    ax[1, 1].grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    P0 = [0,0,0]
    V0 = initial_velocity(speed=76, angle=10.4) 
    W0 = np.array([3, -2545, 5])
    wind = WindField(nx=30, ny=150, nz=50, direction= 45, profile='log', z0=0.03)
    # t1 = solver(P0, V0, W0, wind, dt=0.01)
    # V0 = initial_velocity(speed=46, angle=23.4) 
    # W0 = np.array([3, -9445, 5])
    # t2 = solver(P0, V0, W0, wind, dt=0.01)
    # trajectories = [t1, t2]

    # plot_trajectory(trajectories)
    # plot_coefficients(trajectories)

    # t = []
    # for decay in [0.05, 0.1, 0.2]:
    #     t.append(solver(P0, V0, W0, wind, dt=0.01, decay_rate=decay))
    # plot_trajectory(t)
    t = solver(P0, V0, W0, wind, dt=0.01)
    te = euler_solver(P0, V0, W0, wind, dt=0.01)
    plot_trajectory([t, te])