import numpy as np
from windfield import WindField

def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([np.cos(theta), 0, np.sin(theta)])
    return V0

# Constants for acceleration calculations
r=0.0214; m=0.046; rho=1.204; g=9.81; mu = 1.82e-5 #Tabel B.3 for mu and rho -- # inverse mass for efficiency
A = np.pi * r**2 # cross-sectional area of the ball
constant_property = A * rho/(2 * m) # constant property for efficiency
G = -g * np.array([0, 0, 1]) # gravity vector

def acc(V, W, wind):

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)
    
    cd, cl = coefficients(U_mag, norm(W), r) # get drag and lift coefficients based on current conditions

    # Aerodynamic forces
    D = -constant_property * cd * U_mag * U
    if UW_cross_norm > 0:
        L = constant_property * cl * U_mag**2 * UW_cross / norm(UW_cross) # The "Standard Aerodynamic" Model
    else:
        L = np.array([0, 0, 0]) # No lift if there is no spin or relative velocity

    a = D + L + G # acceleration from Newton's second law

    return a

def coefficients(v, w, r):

    re = (rho * v * (2*r)) / mu # Reynolds Number calculation
    s = (w * r) / v             # Spin Ratio

    # Lift and Drag model based on "Flight Trajectory of a Golf Ball for a Realistic Game" (Kim et al., 2012)
    re_cr = 0.6e5 # Critical Reynolds number for drag crisis (fig 9.22 Brief Introduction to Fluid Mechanics)
    Cd1 = 0.25 ; Cd3 = 0.1e-3 ; Cd2 = (re - re_cr) * Cd3 
    K1 = re * Cd3 ; K2 = re * Cd3 - Cd2

    # ------------ Works only in Re range 0.4e5 to 2.2e5, but gives a more realistic drag crisis behavior ------------
    
    if re < re_cr: #Laminar flow/Transition
        D_re = K1 + Cd1 * K1 - 1
    else: #Turbulent flow
        D_re = K2 + Cd1 * K2 - 0.0225 * Cd2 - 1
    
    cd = 0.2136 * (-2.1 * np.exp(-0.12 * (D_re + s + 0.35)) + 8.9 * np.exp(-0.22 * (D_re + 0.35)))
    
    cl = -0.05 + np.sqrt(0.0025 + 0.36 * s)

    # Printing out Re to check if we are in range
    # print(f"Re: {re:.2e}, Cd: {cd:.3f}, Cl: {cl:.3f}")

    return cd, cl

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
        V.append(V[-1] + acc(V[-1], W[-1], fetch_wind_data(wind, *P[-1])) * dt)
        P.append(P[-1] + V[-1] * dt)
        W.append(W[-1]*np.exp(-5e-4*dt))  # Assuming 4% spin decay every second (4%/60 = 5e-4)
        t.append(t[-1] + dt)
    return np.array(t), np.array(P), np.array(V), np.array(W)



