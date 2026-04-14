import numpy as np
from Tracer.windfield import WindField
import matplotlib.pyplot as plt

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

def acc(V, W, wind):

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)
    
    cd, cl = coefficients(U_mag, norm(W)) # get drag and lift coefficients based on current conditions
    
    # Aerodynamic forces
    D = -constant_property * cd * U_mag * U
    if UW_cross_norm > 0:
        L = constant_property * cl * U_mag**2 * UW_cross / UW_cross_norm # The "Standard Aerodynamic" Model
    else:
        L = np.array([0, 0, 0]) # No lift if there is no spin or relative velocity

    a = D + L + G # acceleration from Newton's second law

    return a

def coefficients(v, w):
    """
    Calculate drag and lift coefficients based on velocity and angular velocity.
    
    Args:
        v: relative velocity magnitude (m/s)
        w: angular velocity magnitude (rad/s)
        eske: boolean indicating which coefficient model to use
    
    Returns:
        (cd, cl): drag and lift coefficients
    """
    re = (v * (2*r)) / mu # Reynolds Number calculation
    s = w * r / v if v > 0 else 0 # Spin parameter
    if s < 0.2:
        cl = -0.05 + np.sqrt(0.0025 + 0.36 * s)
    else:
        cl = 0.2
        
    if re < .4e5:
        cd = 0.46
    else:
        re_c = 0.6e5
        C1 =0.25
        C2 = (re - re_c) * 1e-4
        K1 = re * 1e-4
        K2 = re * 1e-4 - C2
        D = K1 + C1*K1 - 1 if re < re_c else K2 + C1*K2 - 1 - 0.0225*C2
        cd = 0.2136 * (
            - 2.1 * np.exp(-0.12 * (D + A + 0.35))
            + 8.9 * np.exp(-0.22 * (D + 0.35))
            )

    return cd, cl

# Constants from the Slazenger ball study
CD1 = 0.24
CD2 = 0.18
CD3 = 0.06
CL1 = 0.51#0.54
A1 = 90000.0
A2 = 200000.0

def coefficients(v, w):
    """
    Aerodynamic model restricted to driver shots (Re 70k-210k).
    v: velocity (m/s)
    w: angular velocity (rad/s)
    """
    
    # 1. Calculate Reynolds Number and Spin Parameter
    re = (v * (2 * r)) / mu
    s = (w * r) / v
    
    # 2. Lift Coefficient: CL = C_L1 * s^0.4
    cl = CL1 * (max(s, 0)**0.4)
    
    # 3. Drag Coefficient: CD = CD1 + CD2*s + CD3*sin(pi*(Re-A1)/A2)
    # The sine term captures the 'dip' in the drag crisis

    drag_oscillation = CD3 * np.sin(np.pi * (re - A1) / A2)
    cd = CD1 + (CD2 * s) + drag_oscillation
    
    
    return cd, cl

def fetch_wind_data(wind: WindField, x, y, z):
    # This function will fetch the wind data from an API or a file
    # For now, we will just return a dummy wind vector
    x = int(np.clip(x, 0, wind.nx-1))
    y = int(np.clip(y, 0, wind.ny-1))
    z = int(np.clip(z, 0, wind.nz-1))
    return wind.get_point(x, y, z)['velocity']

def solver(P0, V0, W0, wind: WindField, dt=0.01, decay_rate=0.05):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = [0]
    P = [P0] # position array
    V = [V0]
    W = [W0 * 2*np.pi/60] # initial angular velocity (rpm2radps)
    while P[-1][2] >= 0: # while the ball is above the ground
        
        a = acc(V[-1], W[-1], fetch_wind_data(wind, *P[-1]))

        V.append(V[-1] + a * dt)
        P.append(P[-1] + V[-1] * dt)
        W.append(W[-1] * np.exp(-decay_rate * dt))  # Update angular velocity
        t.append(t[-1] + dt)

    return np.array(t), np.array(P), np.array(V), np.array(W)


if __name__ == "__main__":
    # This is where we will run our simulation and plot the results
    # We can change the initial conditions and parameters here to see how they affect the trajectory of the ball
    P0 = [0,0,0]
    V0 = initial_velocity(speed=49.4, angle=25.4) 
    W0 = np.array([3, -2445, 5])
    wind = WindField(nx=30, ny=150, nz=50, direction= 45, profile='log', z0=0.003)


    
    