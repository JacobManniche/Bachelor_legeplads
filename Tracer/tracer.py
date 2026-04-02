import numpy as np
import matplotlib.pyplot as plt


def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([0, np.cos(theta), np.sin(theta)])
    return V0

def acc(V, W, wind, r, m, vc, vs, g, rho):
    pi = np.pi
    cd = drag_coefficient(norm(V), norm(W), r, vc, vs)
    cl = 1.7

    a = -1/(2*m) * cd * pi * r**2 * rho * norm(V-wind) * (V-wind) \
        + cl/m * pi * r**3 * rho * np.cross(W, V)
    
    a[2] += g # add gravity in the z direction

    return a

def spin_ratio(w, r, v):
    return w * r / v

def drag_coefficient(v, w, r, vc, vs): # Based on Mencke et al. 2020
    s = spin_ratio(w, r, v)
    if s>0.05 and v>vc: # at high spin ratios and high velocities, the drag coefficient increases with spin ratio
        return 0.4127*s**0.3056
    else: 
        return 0.2 + 0.346 / (1 + np.exp((v - vc) / vs))

def fetch_wind_data(wind):
    # This function will fetch the wind data from an API or a file
    # For now, we will just return a dummy wind vector
    return wind 

def solver(P0, V0, W0, wind, vc=33, vs=5, r=0.0214, m=0.046, g=-9.81, rho=1.2, dt=0.01):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = np.array([0]) # time array starting at 0
    P = np.array([P0]) # initial position
    V = np.array([V0]) # initial velocity
    W = np.array([W0]) # initial angular velocity
    while P[-1][2]>=0: # integrate until the ball hits the ground
        print(f"Time: {t[-1]:.2f} s, Position: {P[-1]}, Velocity: {V[-1]}, Spin: {W[-1]}", end='\r', flush=True)
        t = np.append(t, t[-1] + dt)
        a = acc(V[-1], W[-1], fetch_wind_data(wind), r, m, vc, vs, g, rho)
        V = np.append(V, [V[-1] + a * dt], axis=0)
        P = np.append(P, [P[-1] + V[-1] * dt], axis=0)

    return t, P