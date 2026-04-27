import numpy as np
import xarray as xr

class Field:
    def __init__(self, ds=None, profile="log", **kwargs):
        """
        ds : xarray.Dataset
            Internal dataset containing coordinates 'x', 'y', 'z' 
            and variables 'U', 'V', 'W', 'tke'.
        profile : str
            Type of wind field to create. Options: 'log', 'uniform', 'rans'.
        kwargs : dict
            Additional parameters for synthetic profiles (direction, U_ref, z0) or RANS dataset
        """

        # make sure profile is lowercase for consistency
        self.profile = profile.lower()
        
        if self.profile == "rans" or ds is not None:
            self.from_rans(ds)
        elif self.profile in ["log", "uniform"]:
            self.synthesize(**kwargs)
        else:
            raise ValueError("Must provide a valid profile type and/or dataset.")

    def synthesize(self, profile="log", x_range=(-250, 250), y_range=(-250, 250), z_height=100, 
                       res=10.0, direction=270, U_ref=10.0, z0=0.03, z_ref = 10.0):
        """Constructor for Log or Uniform profiles."""
        x = np.arange(x_range[0], x_range[1] + res, res)
        y = np.arange(y_range[0], y_range[1] + res, res)
        z = np.arange(1, z_height + 1, 1)
        
        # Create 3D mesh for the vertical profile
        # Velocity only changes with Z in synthetic profiles

        if profile == "uniform":
            z_mag = np.full_like(z, U_ref)
        elif profile == "log":

            # NB: no need for kappa in the calculation since it cancels out
            u_star = U_ref / np.log(z_ref / z0)
            
            z_mag = u_star * np.log(z / z0)

        mag = np.tile(z_mag, (len(x), len(y), 1)) # Shape (nx, ny, nz)
        
        angle = np.radians(direction)

        self.ds = xr.Dataset(
            data_vars={
                'U': (['x', 'y', 'z'], mag * np.cos(angle)),
                'V': (['x', 'y', 'z'], mag * np.sin(angle)),
                'W': (['x', 'y', 'z'], np.zeros_like(mag)),
                'tke': (['x', 'y', 'z'], 0.1 * (mag**2) * np.exp(-z / 50))
            },
            coords={'x': x, 'y': y, 'z': z}
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
        # .interp handles the logic of finding the nearest 'boxes' and weighting them
        point = self.ds.interp(**kwargs)
        
        return np.array([point.U.values, point.V.values, point.W.values])
    
    def get_tke_at(self, **kwargs):
        point = self.ds.interp(**kwargs)
        return float(point.tke.values)
    
    def __repr__(self):
        # Simple representation showing profile type and dataset summary
        return f"Field(profile={self.profile}) \n {self.ds})"

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Create a synthetic log profile
    sim_field = Field(profile="log", direction=0, U_ref=8, z_ref=90, z0=0.03)
    data = sim_field.get_velocity_at(x=0, y=0, z=50.0)
    
    #3. Load from your Cartesian NetCDF
    rans_wind = Field(ds = 'RANS/nc files/flow_flat_2m_2m.nc', profile='rans', scale_factor=8)
    data_rans = rans_wind.get_velocity_at(x=0, y=0, z=50)
    print(f"RANS U: {data_rans[0]:.2f} m/s")