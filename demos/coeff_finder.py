from scipy.optimize import minimize
import numpy as np
from Tracer import initial_velocity, initial_spin_rate, WindField


df_pga_data = np.array([[7.644384e+01, 1.040000e+01, 2.545000e+03, 3.200000e+01, 2.580000e+02],
       [7.242048e+01, 9.300000e+00, 3.663000e+03, 2.900000e+01, 2.280000e+02],
       [6.973824e+01, 9.700000e+00, 4.322000e+03, 3.000000e+01, 2.160000e+02],
       [6.660896e+01, 1.020000e+01, 4.587000e+03, 2.800000e+01, 2.110000e+02],
       [6.482080e+01, 1.030000e+01, 4.404000e+03, 2.700000e+01, 1.990000e+02],
       [6.258560e+01, 1.080000e+01, 4.782000e+03, 2.800000e+01, 1.920000e+02],
       [6.035040e+01, 1.190000e+01, 5.280000e+03, 3.000000e+01, 1.820000e+02],
       [5.811520e+01, 1.400000e+01, 6.204000e+03, 2.900000e+01, 1.720000e+02],
       [5.498592e+01, 1.610000e+01, 7.124000e+03, 3.100000e+01, 1.610000e+02],
       [5.275072e+01, 1.780000e+01, 8.078000e+03, 3.000000e+01, 1.500000e+02],
       [5.006848e+01, 2.000000e+01, 8.793000e+03, 2.900000e+01, 1.390000e+02],
       [4.649216e+01, 2.370000e+01, 9.316000e+03, 2.900000e+01, 1.300000e+02],
        [  63.92672,   12.6    , 2506.     ,   24.     ,  204.     ],
       [  60.3504 ,   11.6    , 2595.     ,   23.     ,  183.     ],
       [  58.1152 ,   12.3    , 4320.     ,   23.     ,  173.     ],
       [  55.88   ,   13.9    , 4504.     ,   23.     ,  163.     ],
       [  52.75072,   13.9    , 4608.     ,   23.     ,  160.     ],
       [  50.96256,   14.6    , 4966.     ,   23.     ,  152.     ],
       [  49.62144,   16.7    , 5904.     ,   23.     ,  142.     ],
       [  47.38624,   18.5    , 6630.     ,   24.     ,  131.     ],
       [  45.59808,   20.8    , 7413.     ,   25.     ,  122.     ],
       [  42.4688 ,   23.5    , 7605.     ,   25.     ,  112.     ],
       [  39.33952,   25.2    , 8465.     ,   25.     ,  101.     ]])

# read csv file
file = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/Confidential/shotdata.csv'
with open(file, 'r') as f:
    lines = f.readlines()
    f.close()

driver_data_row = []

for line in lines[1:]:
    ca, bs, ld, la, sa, sr, mh = [float(l) for l in line.replace(',', '.').split('\t')[1:]]
    
    driver_data_row.append({
        'V0': initial_velocity(bs, la),
        'W0': initial_spin_rate(sr, sa),
        'Target_Height': mh,
        'Target_Carry': ca
    })    


#Script
def norm(arr):
    """Returns the Euclidean norm: sqrt(sum(x_i^2))"""
    return np.sqrt(np.inner(arr, arr))

# Constants for acceleration calculations
r=0.0214; m=0.046; rho=1.204; g=9.81; mu = 1.82e-5 #Tabel B.3 for mu and rho
A = np.pi * r**2 # cross-sectional area of the ball
constant_property = A * rho/(2 * m) # constant property for efficiency
G = -g * np.array([0, 0, 1]) # gravity vector

def acc(V, W, wind, custom_coeffs):

    # Computations
    U = V - wind # relative velocity of the ball with respect to the air
    U_mag = norm(U)
    UW_cross = np.cross(W, U)
    UW_cross_norm = norm(UW_cross)
    
    cd, cl = custom_coeffs(U_mag, norm(W))

    
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
CL1 = 0.51#0.54
A1 = 90000.0
A2 = 200000.0


def fetch_wind_data(wind: WindField, x, y, z):
    # This function will fetch the wind data from an API or a file
    # For now, we will just return a dummy wind vector
    x = int(np.clip(x, 0, wind.nx-1))
    y = int(np.clip(y, 0, wind.ny-1))
    z = int(np.clip(z, 0, wind.nz-1))
    return wind.get_point(x, y, z)['velocity']

def solver(V0, W0, P0=np.array([0, 0, 0]), wind='log', dt=0.01, decay_rate=0.05, custom_coeffs=None):
    # This function will solve the equations of motion for the ball given the initial conditions and parameters
    # It will return the trajectory of the ball as a function of time
    t = [0]
    P = [P0] # position array
    V = [V0]
    W = [W0] # initial angular velocity (rad/s)
    
    if not isinstance(wind, WindField) and wind == 'log' or wind == 'uniform':
        wind = WindField(nx=40, ny=300, nz=50, direction=45, profile=wind, z0=0.003, U_ref=10)

    while P[-1][2] >= 0: # while the ball is above the ground
        
        a = acc(V[-1], W[-1], fetch_wind_data(wind, *P[-1]), custom_coeffs=custom_coeffs)

        V.append(V[-1] + a * dt)
        P.append(P[-1] + V[-1] * dt)
        W.append(W[-1] * np.exp(-decay_rate * dt))  # Update angular velocity
        t.append(t[-1] + dt)

    return np.array(t), np.array(P), np.array(V), np.array(W)



r = 0.0214 # radius of golf ball in meters
mu = 1.81e-5 # dynamic viscosity of air in kg/(m*s)
A1 = 90000.0
A2 = 200000.0

# This is the function the optimizer will try to minimize
def objective_function(params, all_rows):
    cd1, cd2, cd3, cl1 = params
    total_error = 0
    
    # Define the coefficient model using current test parameters
    def test_coeffs(v, w):
        re = (v * (2 * r)) / mu
        s = (w * r) / v if v > 0 else 0
        # Guard against math errors with s**0.4 if spin is negative/zero
        cl = cl1 * (max(s, 0)**0.4) 
        cd = cd1 + (cd2 * s) + cd3 * np.sin(np.pi * (re - A1) / A2)
        return cd, cl

    # Loop through every club (Driver, 3-wood, Irons, etc.)
    for data_row in all_rows:
        t, p, v, w = solver(
            data_row['V0'], 
            data_row['W0'],  
            custom_coeffs=test_coeffs
        )
        
        sim_carry = (p[-1][0]**2 + p[-1][1]**2)**0.5
        sim_max_height = np.max(p[:, 2])
        
        # Calculate squared error for this specific club
        # We weight Carry more heavily than Height (standard practice)
        total_error += (sim_carry - data_row['Target_Carry'])**2
        total_error += (sim_max_height - data_row['Target_Height'])**2
        
    return total_error

# Initial guess from the paper
initial_guess = [0.22, 0.15, 0.50, 0.54] # [CD1, CD2, CD3, CL1]

# Bounds help the optimizer stay within physical reality
# cd1 [0.1 to 0.4], cd2 [0.0 to 0.5], cl1 [0.2 to 0.8], cd3 [0.0 to 0.7]
bounds = [(0.1, 0.4), (0.05, 0.5), (0.2, 0.8), (0.1, 0.8)]

# Run the optimization passing the FULL list of rows
res = minimize(
    objective_function, 
    initial_guess, 
    args=(driver_data_row,), # Note the trailing comma for the tuple!
    bounds=bounds,
    method='L-BFGS-B'
)

print("--- Optimization Complete ---")
print(f"Universal CD1: {res.x[0]:.4f}")
print(f"Universal CD2: {res.x[1]:.4f}")
print(f"Universal CD3: {res.x[2]:.4f}")
print(f"Universal CL1: {res.x[3]:.4f}")


# --- Optimization Complete ---
# Universal CD1: 0.3068
# Universal CD2: 0.3000
# Universal CL1: 0.6464