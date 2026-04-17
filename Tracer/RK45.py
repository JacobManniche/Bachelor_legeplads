from Tracer import WindField
from Tracer.tracer import fetch_wind_data, acc

import numpy as np
from scipy.integrate import solve_ivp

def odesystem(t, y, W_initial, wind, decay_rate):
    # 1. Unpack the state
    P = y[:3]
    V = y[3:]
    
    # 2. Update Spin (Decay is time-dependent)
    W = W_initial * np.exp(-decay_rate * t)
    
    # 3. Get Acceleration
    # (Assuming fetch_wind_data and acc are accessible)
    wind_vec = fetch_wind_data(wind, *P)
    a = acc(V, W, wind_vec)
    
    # 4. Return derivatives: [velocity_x, y, z, acceleration_x, y, z]
    return np.concatenate([V, a])

# Define the 'Ground Hit' event to stop the solver early

def hit_ground(t, y, *args):
    return y[2] # The Z-coordinate
hit_ground.terminal = True
hit_ground.direction = -1

def solver_rk45(V0, W0, P0=np.array([0, 0, 0]), wind=None, dt=0.05, decay_rate=0.05, mt=15, rtol=1e-5):
    # Similar setup as above but with fixed time steps
    # Initial state: Combine P0 and V0
    y0 = np.concatenate([P0, V0])

    if wind is None:
        wind = WindField(nx=300, ny=50, nz=50, direction=45, profile='log', z0=0.003, U_ref=6) # default wind field if none provided

    t_requested = np.arange(0, mt + dt, dt) if dt else None

    sol = solve_ivp(
        odesystem, 
        t_span=(0, mt),       # Max time 15 seconds
        y0=y0, 
        method='RK45',        # Dormand-Prince adaptive solver
        args=(W0, wind, decay_rate),
        events=hit_ground,    # Stops the moment Z hits 0
        t_eval=t_requested,
        rtol=rtol, atol=rtol/1000  # Precision settings
    )

    # Accessing results:
    t_steps = sol.t
    positions = sol.y[:3, :].T # Shape (N, 3)
    velocities = sol.y[3:, :].T
    spin_rates = W0 * np.exp(-decay_rate * t_steps[:, np.newaxis])

    return t_steps, positions, velocities, spin_rates

