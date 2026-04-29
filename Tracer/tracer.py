import numpy as np
from Tracer.windfield import WindField

def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([np.cos(theta), 0, np.sin(theta)])
    return V0

def initial_spin_rate(spin_rate, spin_axis=0):
    """Returns the initial spin rate in radians per second given the spin rate in rpm"""
    phi = np.radians(spin_axis)
    
    # Initial Spin (W0) in rad/s
    w_mag = spin_rate * (2 * np.pi / 60)
    
    # For X-axis flight, pure backspin is around the Y-axis [0, -1, 0]
    # We tilt that Y-axis by the 'spin_axis' angle
    w0 = w_mag * np.array([
        0,               # X-component
        -np.cos(phi),   # Y-component (Backspin)
        np.sin(phi),    # Z-component (Sidespin)
    ])
    
    return w0

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

    # The "Standard Aerodynamic" Model
    D = -constant_property * cd * U_mag * U
    
    # The "Standard Aerodynamic" Model
    L = constant_property * cl * U_mag**2 * UW_cross / UW_cross_norm if UW_cross_norm > 0 else np.array([0, 0, 0])
    
    a = D + L + G # acceleration from Newton's second law

    return a


# Constants from the Slazenger ball study
CD1 = 0.24
CD2 = 0.18
CD3 = 0.06
CL1 = 0.54
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
    s = (w * r) / v if v > 0 else 0  # spin parameter, avoid division by zero
    
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

def solver(V0, W0, P0=np.array([0, 0, 0]), wind=None, dt=0.01, decay_rate=0.05):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = [0]
    P = [P0] # position array
    V = [V0]
    W = [W0] # initial angular velocity (rad/s)

    if wind is None:
        wind = WindField(nx=300, ny=50, nz=50, direction=45, profile='log', z0=0.003, U_ref=6) # default wind field if none provided

    while P[-1][2] >= 0: # while the ball is above the ground
        
        a = acc(V[-1], W[-1], fetch_wind_data(wind, *P[-1]))

        V.append(V[-1] + a * dt)
        P.append(P[-1] + V[-1] * dt)
        W.append(W[-1] * np.exp(-decay_rate * dt))  # Update angular velocity
        t.append(t[-1] + dt)

    return np.array(t), np.array(P), np.array(V), np.array(W)


if __name__ == "__main__":
    import Tracer.debug_tools as d
    V0 = initial_velocity(speed=49.4, angle=25.4) 
    W0 = initial_spin_rate(spin_rate=2540, spin_axis=1.25)

    wind = WindField(nx=60, ny=150, nz=50, direction=45, profile='log')
    
    traces = [solver(V0, W0, wind=wind, dt=0.01)]
    d.plot_trajectories(traces)
    d.plot_coefficients(traces)
    