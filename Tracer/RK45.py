from Tracer.field import Field
from Tracer.tracer import acc

import numpy as np
from scipy.integrate import solve_ivp

def odesystem(t, y, W_initial, field, decay_rate):
    # 1. Unpack the state
    P = y[:3]
    V = y[3:]
    
    # 2. Update Spin (Decay is time-dependent)
    W = W_initial * np.exp(-decay_rate * t)
    
    # 3. Get Acceleration
    # (Assuming fetch_wind_data and acc are accessible)
    field_vec = field.get_velocity_at(*P)
    a = acc(V, W, field_vec)
    
    # 4. Return derivatives: [velocity_x, y, z, acceleration_x, y, z]
    return np.concatenate([V, a])

# Define the 'Ground Hit' event to stop the solver early

def hit_ground(t, y, *args):
    return y[2] # The Z-coordinate
hit_ground.terminal = True
hit_ground.direction = -1

def solver_rk45(V0, W0, P0=np.array([0, 0, 0]), field=None, dt=0.05, decay_rate=0.05, mt=15, rtol=1e-6):
    # Similar setup as above but with fixed time steps
    # Initial state: Combine P0 and V0
    y0 = np.concatenate([P0, V0])

    if field is None:
        field = Field(direction=45, profile='log', z0=0.003, U_ref=6) # default wind field if none provided

    t_requested = np.arange(0, mt + dt, dt) if dt else None

    sol = solve_ivp(
        odesystem, 
        t_span=(0, mt),       # Max time 15 seconds
        y0=y0, 
        method='RK45',        # Dormand-Prince adaptive solver
        args=(W0, field, decay_rate),
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

if __name__ == "__main__":
    from Tracer import initial_spin_rate, initial_velocity
    from Tracer.debug_tools import plot_trajectories
    # Example usage
    V0 = initial_velocity(speed=76.44384, angle=10.4)  # Initial velocity vector
    W0 = initial_spin_rate(spin_rate=2545, spin_axis=1.25)  # Initial spin vector
    wind = Field(ds='RANS/flow_flat_4m_2m.nc', profile='rans') # Example wind field
    print(wind.get_velocity_at(10,10,12))
    #plot_trajectories([solver_rk45(V0, W0, field=wind)])