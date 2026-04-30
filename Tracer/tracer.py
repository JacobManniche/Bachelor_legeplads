import numpy as np
from Tracer.windfield import WindField
from Tracer.solvers import solver_rk45, solver_euler

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

class Trajectory:
    def __init__(self, ball_speed, launch_angle, spin_rate, spin_axis=0, P0=np.array([0, 0, 0]), wind=WindField(profile='log')):
        """
        Initializes the Trajectory object with initial conditions and wind field.
        Parameters:
        - ball_speed: Initial speed of the ball in m/s
        - launch_angle: Launch angle of the ball in degrees
        - spin_rate: Spin rate of the ball in rpm
        - spin_axis: Spin axis angle in degrees (default: 0 for pure backspin)
        - P0: Initial position vector (m)
        - wind: Wind field object that provides the get_velocity_at(x, y, z)
        """
        self.P0 = P0
        self.V0 = initial_velocity(speed=ball_speed, angle=launch_angle)
        self.W0 = initial_spin_rate(spin_rate=spin_rate, spin_axis=spin_axis)
        self.wind = wind

    def solve(self, solver='rk45', dt=0.01, **kwargs):
        """
        Solves the trajectory of the ball using the specified solver method.
        Parameters:
        - solver: 'rk45' for Runge-Kutta 4(5) method, 'euler' for simple Euler method.
        - dt: Time step for the solver (only visual for rk45 can be set to None).
        - kwargs: Additional parameters to pass to the solver 
            decay_rate: Decay rate for spin (default: 0.05)
            r_tol: Relative tolerance for RK45 (default: 1e-6)
            mt: Max time for rk45 solver (default: 15 seconds)
        Returns:
        - t: Array of time steps
        - p: Array of positions at each time step
        - v: Array of velocities at each time step
        - w: Array of spin rates at each time step
        """
        if solver == 'rk45':
            t, p, v, w = solver_rk45(self.V0, self.W0, P0=self.P0, wind=self.wind, dt=dt, **kwargs)
        elif solver == 'euler':
            t, p, v, w = solver_euler(self.V0, self.W0, P0=self.P0, wind=self.wind, dt=dt, **kwargs)
        else:
            raise ValueError("Invalid solver specified. Use 'rk45' or 'euler'.")
        
        self.t = t
        self.p = p
        self.v = v
        self.w = w
        self.traj = (t, p, v, w)

        return t, p, v, w

    def plot(self):
        from Tracer.debug_tools import plot_trajectories
        plot_trajectories([self])



if __name__ == "__main__":
    # Example usage
    file = '/Users/eskefr/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/6. semester/Bachelor/Github/Bachelor_legeplads/RANS/nc files/flowdata_2m_cartesian.nc'
    wind = WindField(profile='rans', ds=file)
    trajectory = Trajectory(ball_speed=76.44384, launch_angle=10.4, spin_rate=2545, spin_axis=1.25)
    trajectory.solve(solver='euler', dt=0.01)
    trajectory.plot()