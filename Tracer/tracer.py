import numpy as np

def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

def initial_velocity(speed, angle):
    """Returns the initial velocity vector given the speed and angle of projection"""
    theta = np.radians(angle)
    V0 = speed * np.array([0, np.cos(theta), np.sin(theta)])
    return V0

def acc(V, W, wind):

    # Constants
    r=0.0214; m_inv=21.7; rho=1.2; g=9.81 # inverse mass for efficiency

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    A = np.pi * r**2 # cross-sectional area of the ball
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)
    
    cd, cl = coefficients(U_mag, norm(W), r) # get drag and lift coefficients based on current conditions

    # Aerodynamic forces
    D = -.5 * m_inv * cd * A * rho * U_mag * U
    if UW_cross_norm > 0:
        L = .5 * m_inv * cl * A * rho * U_mag**2 * UW_cross / norm(UW_cross) # The "Standard Aerodynamic" Model
    else:
        L = np.array([0, 0, 0])

    G = -g * np.array([0, 0, 1]) # gravity vector

    a = D + L + G # acceleration from Newton's second law

    return a

def coefficients(v, w, r):
    ## TODO add more realistic models for the drag and lift coefficients

    re = (1.2 * v * (2*r)) / 1.78e-5 # Reynolds Number calculation
    s = (w * r) / v                   # Spin Ratio
    
    # Baseline Drag based on Reynolds (simplified transition)
    if re < 40000:
        cd_base = 0.5
    elif re < 150000:
        # Linear drop during drag crisis
        cd_base = 0.5 - 0.28 * ((re - 40000) / 110000)
    else:
        cd_base = 0.22
        
    # Add Spin-Induced Drag (Mencke/Lieberman logic)
    cd = cd_base + 0.3 * (s**2)
    cl =  0.1 + 0.5 * s
    return cd, cl

def fetch_wind_data(wind):
    # This function will fetch the wind data from an API or a file
    # For now, we will just return a dummy wind vector
    return wind

def solver(P0, V0, W0, wind, dt=0.01, max_time=15):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    max_steps = int(max_time / dt)
    
    t = np.arange(0, max_time, dt) # time array starting at 0
    P = np.zeros((max_steps, 3)) # position array
    V = np.zeros((max_steps, 3)) # velocity array
    W = np.zeros((max_steps, 3)) # initial angular velocity (rpm2radps)
    P[0] = P0
    V[0] = V0
    W[0] = W0 * 2*np.pi/60  # Convert rpm to rad/s
    for i in range(1, max_steps):
        a = acc(V[i-1], W[i-1], fetch_wind_data(wind))
        V[i] = V[i-1] + a * dt
        P[i] = P[i-1] + V[i-1] * dt
        W[i] = W[i-1]*np.exp(-5e-4*dt)  # Assuming 4% spin decay every second (4%/60 = 5e-4)
        if P[i][2] < 0: # Stop if the ball hits the ground
            P = P[:i+1]
            t = t[:i+1]
            break
    return t, P

