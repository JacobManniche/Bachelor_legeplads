import numpy as np
from Tracer.lookup import get_cd, get_cl
from Tracer.windfield import WindField

def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([np.cos(theta), 0, np.sin(theta)])
    return V0

# Constants for acceleration calculations
r=0.0214; m=0.046; rho=1.204; g=9.81; mu = 1.82e-5 #Tabel B.3 for mu and rho
A = np.pi * r**2 # cross-sectional area of the ball
constant_property = A * rho/(2 * m) # constant property for efficiency
G = -g * np.array([0, 0, 1]) # gravity vector
I = (2/5) * m * r**2 # moment of inertia for a solid sphere

def acc(V, W, wind):

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)
    
    cd, cl, ct = coefficients(U_mag, norm(W)) # get drag and lift coefficients based on current conditions
    
    # Aerodynamic forces
    D = -constant_property * cd * U_mag * U
    if UW_cross_norm > 0:
        L = constant_property * cl * U_mag**2 * UW_cross / norm(UW_cross) # The "Standard Aerodynamic" Model
    else:
        L = np.array([0, 0, 0]) # No lift if there is no spin or relative velocity
    # a becomes nan !!!!!
    a = D + L + G # acceleration from Newton's second law
    
    # Calculate angular acceleration

    W_mag = norm(W)
    if W_mag > 1: # Threshold to stop calculation when spin is near zero
        unit_W = W / W_mag
        # Torque magnitude opposes rotation
        torque_mag = -.5* rho * r**5 * W_mag**2 * ct
        alp = (torque_mag / I) * unit_W
    else:
        alp = np.zeros(3)

    return a, alp

def coefficients(v, w):
    """
    Calculate drag and lift coefficients based on velocity and angular velocity.
    
    Args:
        v: relative velocity magnitude (m/s)
        w: angular velocity magnitude (rad/s)
    
    Returns:
        (cd, cl): drag and lift coefficients
    """
    
    re = (v * (2*r)) / mu # Reynolds Number calculation
    s = w * r / v if v > 0 else 0 # Spin parameter

    #cl = 0.686517 * s + 0.074215 # Linear fit for CL from fit.py
    
    #cd = 0.000000 * s**2 + 0.000000 * s + 0.295000 # Quadratic fit for CD from fit.py
    
    if re < 6.5e4:
        cd = 0.7
    elif re < 7.5e4:
        cd = 1.29e-10 * re**2 - 2.59e-05 * re + 1.5
    elif re < 1.5e5:
        cd = 1.91e-11*re**2 - 5.40e-06*re + 0.56
    else:
        cd = 0.2
    
    if s < 0.3:
        cl = -3.25*s**2 + 1.99*s
    else:
        cl = 0.2

    if re < 360:
        ct = 128.9/re
    elif re < 68e3:
        ct = 6.7545/re**.5
    else:
        ct = 0.2398/re**0.2
    return cd, cl, ct

def fetch_wind_data(wind: WindField, x, y, z):
    # This function will fetch the wind data from an API or a file
    # For now, we will just return a dummy wind vector
    x = int(np.clip(x, 0, wind.nx-1))
    y = int(np.clip(y, 0, wind.ny-1))
    z = int(np.clip(z, 0, wind.nz-1))
    return wind.get_point(x, y, z)['velocity']

def solver(P0, V0, W0, wind: WindField, dt=0.01):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = [0]
    P = [P0] # position array
    V = [V0]
    W = [W0 * 2*np.pi/60] # initial angular velocity (rpm2radps)
    while P[-1][2] >= 0: # while the ball is above the ground
        
        a, alp = acc(V[-1], W[-1], fetch_wind_data(wind, *P[-1]))
        V.append(V[-1] + a * dt)
        P.append(P[-1] + V[-1] * dt)
        #W.append(W[-1] + alp * dt)  # Update angular velocity
        W.append(W[-1] * np.exp(-0.5 * dt))  # Update angular velocity
        t.append(t[-1] + dt)

    return np.array(t), np.array(P), np.array(V), np.array(W)

if __name__ == "__main__":
    # This is where we will run our simulation and plot the results
    # We can change the initial conditions and parameters here to see how they affect the trajectory of the ball
    P0 = [0,0,0]
    #V0 = initial_velocity(speed=48, angle=60) 
    V0 = initial_velocity(speed=39.4, angle=25.4) 
    W0 = np.array([3, -8445, 5])
    wind = WindField(nx=30, ny=150, nz=50, direction= -90, profile='log', z0=0.03)
    #wind = WindField(nx=30, ny=150, nz=50, profile='uniform', U_ref=0)

    t, p, v, w = solver(P0, V0, W0, wind)
    print(f"Final position: {p[-1][0]-p[0][0]}")

    import matplotlib.pyplot as plt
    ax = plt.subplots(3, 1, figsize=(10, 8))[1]
    ax[0].plot(p[:, 0], p[:, 2]) # Plot height vs time
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Height (m)')
    ax[0].set_title('Trajectory of the Golf Ball')
    ax[0].grid()
    ax[1].plot(p[:, 0], p[:, 1]) # Plot horizontal distance vs time
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Horizontal Distance (m)')
    ax[1].set_title('Horizontal Trajectory of the Golf Ball')
    ax[1].grid()
    ax[2].plot(t, w[:, 1], label='Spin Rate (rad/s)')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Spin Rate (rad/s)')
    ax[2].set_title('Spin Rate Decay Over Time')
    ax[2].grid()
    ax[2].legend()
    plt.tight_layout()
    plt.show()