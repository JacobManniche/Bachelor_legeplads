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
    
    # For X-axis flight; X: rifle, Y: Backspin, Z: Sidespin 
    w0 = w_mag * np.array([0,-np.cos(phi), np.sin(phi)])

    return w0
class Trajectory:
    def __init__(self, ball_speed, launch_angle, spin_rate, spin_axis=0, orientation=0, P0=np.array([0, 0, 0]), wind=None, fluc=None):
        """
        Initializes the Trajectory object with initial conditions and wind field.
        Parameters:
        - ball_speed: Initial speed of the ball in m/s
        - launch_angle: Launch angle of the ball in degrees
        - spin_rate: Spin rate of the ball in rpm
        - spin_axis: Spin axis angle in degrees (default: 0 for pure backspin)
        - P0: Initial position vector (m)
        - orientation: Orientation angle in degrees (default: 0)
        - wind: WindField object that provides the get_velocity_at(x, y, z)
        - fluc: Fluctuator object that provides the get_fluctuation_at(pos, tke, epsilon) method
        """
        self.args = (ball_speed, launch_angle, spin_rate, spin_axis, orientation)
        self.P0 = P0
        self.V0 = initial_velocity(speed=ball_speed, angle=launch_angle)
        self.W0 = initial_spin_rate(spin_rate=spin_rate, spin_axis=spin_axis)
        if orientation != 0:
            self.rotate(orientation)
        if wind is None:
            print("No wind provided. Using default uniform wind with U_ref=0.")
            wind = WindField(profile='uniform', U_ref=0)
        self.wind = wind
        self.fluc = fluc

        self.is_solved = False

    def rotate(self, angle):
        """Rotates the initial velocity and spin vectors by a given angle in degrees around the Z-axis."""
        psi = np.radians(angle)
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0,            0,           1]])
        self.V0 = Rz @ self.V0
        self.W0 = Rz @ self.W0

        self.is_solved = False
        
        return self

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
            if self.fluc and self.fluc.method in ['ou', 'langevin', 'simple']:
                raise ValueError("RK45 solver is not compatible with temporal stochastic turbulence methods (OU, Langevin, simple). Please use 'euler' solver for these methods.")
            t, p, v, w = solver_rk45(self.V0, self.W0, P0=self.P0, wind=self.wind, fluc=self.fluc, dt=dt, **kwargs)
        elif solver == 'euler':
            t, p, v, w = solver_euler(self.V0, self.W0, P0=self.P0, wind=self.wind, fluc=self.fluc, dt=dt, **kwargs)
        else:
            raise ValueError("Invalid solver specified. Use 'rk45' or 'euler'.")
        
        self.t = t
        self.p = p
        self.v = v
        self.w = w
        self.traj = (t, p, v, w)

        self.is_solved = True

        return t, p, v, w

    def plot(self):
        from Tracer.debug_tools import plot_trajectories
        plot_trajectories([self])

    def animate(self):
        from Tracer.animate import animate
        animate(self)

    def __repr__(self):
        if self.is_solved:
            return f"Trajectory(ball_speed={self.args[0]}, launch_angle={self.args[1]}, spin_rate={self.args[2]}, spin_axis={self.args[3]}) \nV0={self.V0.round(2)}, \nW0={self.W0.round(2)}, \nP0={self.P0.round(2)}, \nFinal Position={self.p[-1].round(2)}, \n Time={self.t[-1]:.2f} s"
        else:
            return f"Trajectory(V0={self.V0.round(2)}, W0={self.W0.round(2)}, P0={self.P0.round(2)})"


if __name__ == "__main__":
    from Tracer.fluctuator import Fluctuator
    # Example usage
    wind = WindField(profile='log', U_ref=8, z_ref=10, z0=0.03, direction=45)

    landing_points = {method: [] for method in ['simple', 'Langevin', 'POD']}

    for method in ['simple', 'Langevin', 'POD']:
        for i in range(10):
            fluc = Fluctuator(method=method, dt=0.01, C0=2.1, cf=1.0)
            traj = Trajectory(
                ball_speed=76, 
                launch_angle=13, 
                spin_rate=2500,
                wind=wind,
                fluc=fluc
            )
            if method in ['Langevin', 'simple']:
                traj.solve(solver='euler')
            else:
                traj.solve('rk45')
            landing_points[method].append(traj.p[-1][:2])
