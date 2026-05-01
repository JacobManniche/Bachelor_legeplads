import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class WindField:
    def __init__(self, profile="log", ds=None, **kwargs):
        """
        profile : str
            Type of wind field to create. Options: 'log', 'uniform', 'rans'.
        ds : xarray.Dataset
            RANS dataset containing coordinates 'x', 'y', 'z'
            and variables 'U', 'V', 'W', 'tke'.
        kwargs : dict
            Additional parameters for synthetic profiles (direction, U_ref, z0)
        """

        # make sure profile is lowercase for consistency
        self.profile = profile.lower()
        
        if self.profile == "rans" or ds is not None:
            # If a dataset is provided, we use U_ref as a scaling factor to adjust the wind speeds, defaulting to 8 m/s if not specified
            self.from_rans(ds, scale_factor=kwargs.get('U_ref', 8.0))

        elif self.profile in ["log", "uniform"]:
            self.synthesize(**kwargs)

        else:
            raise ValueError("Must provide a valid profile type and/or dataset.")

        self._setup_interpolators()
        
    def _setup_interpolators(self):
        if self.profile == 'uniform':
            # Store static vector for O(1) lookup
            self._static_wind = np.array([self.ds.U.values[0], self.ds.V.values[0], self.ds.W.values[0]])
            return

        # Extract coordinates and data for interpolation
        coords = [self.ds[dim].values for dim in self.ds.dims]
        
        # Combined U, V, W into one array for a single lookup pass
        combined_data = np.stack([self.ds.U.values, self.ds.V.values, self.ds.W.values, self.ds.tke.values, self.ds.epsilon.values], axis=-1)
        
        self.interpolator = RegularGridInterpolator(
            coords, 
            combined_data, 
            method='linear', 
            bounds_error=False, 
            fill_value=0 
        )

    def synthesize(self, z_height=100, direction=0, U_ref=10.0, z0=0.03, z_ref = 10.0):
        """Constructor for Log or Uniform profiles."""
        # Generate vertical grid (logarithmically spaced for better resolution near the ground)
        # always start at 0.1m to avoid log(0) issues, and go up to 100m which is a typical max height for golf ball trajectories
        
        # Create 3D mesh for the vertical profile
        # Velocity only changes with Z in synthetic profiles

        if self.profile == "uniform":
            z = np.array([z_ref])
            z_mag = np.array([U_ref])
        elif self.profile == "log":

            z = np.logspace(-1, np.log10(z_height), num=z_height)
            # NB: no need for kappa in the calculation since it cancels out
            kappa = 0.4
            Cmu = 0.09
            u_star = U_ref / np.log(z_ref / z0)
            z_mag = u_star * np.log(z / z0)
        
        angle = np.radians(direction)

        self.ds = xr.Dataset(
            data_vars={
                'U': (['z'], z_mag * np.cos(angle)),
                'V': (['z'], z_mag * np.sin(angle)),
                'W': (['z'], np.zeros_like(z_mag)),
                'tke': (['z'], np.full_like(z, u_star**2 / np.sqrt(Cmu))),
                'epsilon': (['z'], u_star**3 / (kappa * z))
            },
            coords={'z': z}
        )

    def from_rans(self, path_or_ds, scale_factor=8):
        """Constructor for existing Cartesian RANS NetCDF files."""
        if isinstance(path_or_ds, str):
            ds = xr.open_dataset(path_or_ds)
        else:
            ds = path_or_ds

        ds['U'] *= scale_factor
        ds['V'] *= scale_factor
        ds['W'] *= scale_factor
        self.ds = ds

    def get_velocity_at(self,**kwargs):
        """
        Interpolates the wind field at an arbitrary spatial point.
        Returns a dict with velocity vector and tke.
            - kwargs should be in the form of (x=..., y=..., z=...)
            - method can be 'linear', 'nearest', etc. (xarray interpolation methods)
        """
        # if uniform always return the same value, no need to interpolate
        if self.profile == 'uniform':
            return self._static_wind
        
        # for log profile we only need to interpolate in the vertical direction since it's horizontally uniform
        elif self.profile == 'log':
            return self.interpolator([kwargs['z']])[0, :3]
        
        # for RANS we need to interpolate in all three dimensions, xarray's .interp can handle that
        elif self.profile == 'rans':
            return self.interpolator([kwargs['x'], kwargs['y'], kwargs['z']])[0, :3]
    
    def get_tke_at(self, **kwargs):
        return float(self.interpolator([kwargs['x'], kwargs['y'], kwargs['z']])[0, 3])
    
    def get_epsilon_at(self, **kwargs):
        return float(self.interpolator([kwargs['x'], kwargs['y'], kwargs['z']])[0, 4])

    def __repr__(self):
        # Simple representation showing profile type and dataset summary
        return f"WindField(profile={self.profile}) \n {self.ds})"

# --- Usage Example ---
if __name__ == "__main__":
    from Tracer import Trajectory
    # 1. Create a synthetic log profile
    sim_field = WindField(profile="log", direction=0, U_ref=8, z_ref=90, z0=0.03)

    cond = {'ball_speed': 76.4, 'launch_angle': 10.4, 'spin_rate': 2545, 'spin_axis': 1.25}

    traj = Trajectory(**cond, wind=sim_field)
    traj.solve()
    traj.plot()

    #3. Load from your Cartesian NetCDF
    rans_wind = WindField(ds = '../RANS/nc files/flowdata_2m_cartesian.nc', profile='rans', U_ref=8)
    traj = Trajectory(**cond, wind=rans_wind)
    traj.solve()
    traj.plot()
    
